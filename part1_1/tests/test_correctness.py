"""Fast CPU checks, including mathematical algorithm equivalence."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from part1_1.data import ClientData, split_indices, validate_arrays
from part1_1.federation import Client, ClientUpdate, aggregate, evaluate_global, select_clients
from part1_1.model import clone_state, initial_model
from part1_1.runner import Config, run_experiment, state_digest
from part1_1.utils import load_checkpoint, setup


def aggregation_check():
    states = [dict(w=torch.tensor([1., -2., 3.])), dict(w=torch.tensor([4., 6., -1.])),
              dict(w=torch.tensor([-3., 2., 9.]))]
    counts = [1, 3, 6]
    actual = aggregate([ClientUpdate(i, n, s) for i, (n, s) in enumerate(zip(counts, states))])["w"]
    manual = torch.tensor([(1 + 12 - 18) / 10, (-2 + 18 + 12) / 10, (3 - 3 + 54) / 10])
    return dict(max_abs_error=float((actual - manual).abs().max()), tolerance=1e-6,
                actual=actual.tolist(), expected=manual.tolist())


def centralized_equivalence():
    setup(7)
    g = torch.Generator().manual_seed(8)
    x = torch.randn(17, 28, 28, generator=g)
    y = torch.randint(0, 62, (17,), generator=g)
    factory = lambda: initial_model(7)
    central = factory()
    broadcast = clone_state(central)
    updates, start = [], 0
    for cid, n in enumerate([2, 5, 10]):
        c = Client(ClientData(cid, x[start:start+n], y[start:start+n], x[:0], y[:0]), factory)
        updates.append(c.train(broadcast, 1, .03, n, 9))
        start += n
    fed = aggregate(updates)
    opt = torch.optim.SGD(central.parameters(), lr=.03)
    opt.zero_grad()
    nn.functional.cross_entropy(central(x), y).backward()
    opt.step()
    error = max(float((fed[k] - v).abs().max()) for k, v in central.state_dict().items())
    return dict(max_abs_error=error, tolerance=3e-7, client_sizes=[2, 5, 10], lr=.03,
                explanation="One full-batch local step; mean loss gradients weighted by training counts")


class CorrectnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup(7)

    def test_exact_aggregation(self):
        r = aggregation_check()
        self.assertLessEqual(r["max_abs_error"], r["tolerance"])

    def test_centralized_full_batch_equivalence(self):
        r = centralized_equivalence()
        self.assertLessEqual(r["max_abs_error"], r["tolerance"])

    def test_tiny_disjoint_splits(self):
        for n in [0, 1, 2, 3, 4, 5, 118, 393]:
            tr, va = split_indices(n, 10)
            self.assertEqual(len(set(tr) & set(va)), 0)
            self.assertEqual(sorted([*tr, *va]), list(range(n)))
            np.testing.assert_array_equal(tr, split_indices(n, 10)[0])
            if n >= 2:
                self.assertTrue(len(tr) and len(va))
        self.assertEqual([len(i) for i in split_indices(1, 0)], [1, 0])

    def test_sampling(self):
        for frac, k in [(.02, 2), (.05, 5), (.1, 10), (.001, 1)]:
            a = select_clients(list(range(100)), frac, np.random.default_rng(8))
            b = select_clients(list(range(100)), frac, np.random.default_rng(8))
            self.assertEqual(len(a), k)
            self.assertEqual(len(set(a)), k)
            self.assertEqual(a, b)

    def test_reject_bad_updates(self):
        for n in [0, -1]:
            with self.assertRaises(ValueError):
                aggregate([ClientUpdate(0, n, dict(w=torch.ones(2)))])
        for bad in [float("nan"), float("inf")]:
            with self.assertRaises(FloatingPointError):
                aggregate([ClientUpdate(0, 1, dict(w=torch.tensor([bad])))])
        with self.assertRaises(ValueError):
            aggregate([])
        with self.assertRaises(ValueError):
            aggregate([ClientUpdate(0, 1, dict(w=torch.ones(2)))] * 2)
        with self.assertRaises(ValueError):
            aggregate([ClientUpdate(0, 1, dict(w=torch.ones(2))), ClientUpdate(1, 2, dict(w=torch.ones(3)))])

    def test_broadcast_isolation_and_fresh_optimizer(self):
        f = lambda: initial_model(5)
        x, y = torch.randn(7, 28, 28), torch.arange(7)
        c = Client(ClientData(0, x, y, x[:0], y[:0]), f)
        state = clone_state(f())
        digest = state_digest(state)
        a = c.train(state, 2, .01, 3, 22)
        b = c.train(state, 2, .01, 3, 22)
        self.assertEqual(digest, state_digest(state))
        self.assertEqual(state_digest(a.state), state_digest(b.state))

    def test_empty_client_and_missing_classes(self):
        x, y = torch.zeros(1, 28, 28), torch.zeros(1, dtype=torch.long)
        c = Client(ClientData(0, x, y, x[:0], y[:0]), lambda: initial_model(0))
        self.assertEqual(c.evaluate(initial_model(0), "validation")["n"], 0)
        c.train(clone_state(initial_model(0)), 1, .01, 32, 0)
        empty = Client(ClientData(1, x[:0], y[:0], x[:0], y[:0]), lambda: initial_model(0))
        with self.assertRaises(ValueError):
            empty.train(clone_state(initial_model(0)), 1, .01, 32, 0)

    def test_sample_weighted_metrics(self):
        model = initial_model(1)
        for p in model.parameters():
            p.data.zero_()
        clients = {}
        for cid, (n, label) in enumerate([(1, 0), (9, 1)]):
            x, y = torch.zeros(n, 28, 28), torch.full((n,), label, dtype=torch.long)
            clients[cid] = Client(ClientData(cid, x, y, x, y), lambda: initial_model(1))
        metrics, _ = evaluate_global(model, clients)
        self.assertAlmostEqual(metrics["validation_accuracy"], .1)
        self.assertAlmostEqual(metrics["validation_loss"], np.log(62), places=6)

    def test_invalid_pixels_labels_and_configuration(self):
        with self.assertRaises(ValueError):
            validate_arrays(np.full((1, 28, 28), np.nan), np.array([0]))
        with self.assertRaises(ValueError):
            validate_arrays(np.zeros((1, 28, 28)), np.array([62]))
        with self.assertRaises(ValueError):
            Config(fraction=.11).validate()

    def test_empty_source_arrays(self):
        x, y = validate_arrays([], [])
        self.assertEqual(x.shape, (0, 28, 28))
        self.assertEqual(y.shape, (0,))
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(y.dtype, np.int64)

    def test_extreme_training_count_imbalance(self):
        updates = [ClientUpdate(0, 1, dict(w=torch.tensor([0.]))),
                   ClientUpdate(1, 99999, dict(w=torch.tensor([10.])))]
        self.assertAlmostEqual(float(aggregate(updates)["w"]), 9.9999, places=5)

    def test_nonfinite_local_loss_rejected(self):
        x, y = torch.full((2, 28, 28), float("nan")), torch.zeros(2, dtype=torch.long)
        c = Client(ClientData(0, x, y, x[:0], y[:0]), lambda: initial_model(0))
        with self.assertRaises(FloatingPointError):
            c.train(clone_state(initial_model(0)), 1, .01, 32, 0)

    def test_resume_is_identical(self):
        x, y = torch.randn(12, 28, 28), torch.arange(12) % 2
        data = {i: ClientData(i, x[i:i+3], y[i:i+3], x[9:12], y[9:12]) for i in range(10)}
        config = Config(rounds=3, lr=.01, seed=5)
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d)/"complete", Path(d)/"resumed"
            run_experiment(data, config, a)
            run_experiment(data, config, b, stop_after=1)
            run_experiment(data, config, b)
            pa, pb = load_checkpoint(a/"last.pt"), load_checkpoint(b/"last.pt")
            self.assertEqual(state_digest(pa["state"]), state_digest(pb["state"]))
            self.assertEqual(pa["selections"], pb["selections"])


if __name__ == "__main__":
    unittest.main()
