"""Run the existing CPU correctness tests on the server and save machine-readable evidence."""
import io
from pathlib import Path
import time
import unittest

from .utils import write_json


def main():
    root = Path("part1_1_deepdish5_outputs/validation")
    root.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover("part1_1/tests")
    start = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    log = stream.getvalue()
    print(log, flush=True)
    (root/"correctness_tests.log").write_text(log, encoding="utf-8")
    write_json(root/"correctness_tests.json", dict(tests_run=result.testsRun,
               failures=len(result.failures), errors=len(result.errors), successful=result.wasSuccessful(),
               elapsed_seconds=time.perf_counter()-start))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
