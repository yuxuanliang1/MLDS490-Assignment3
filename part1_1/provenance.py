"""Record source/input integrity and create a concise navigable artifact index."""
from pathlib import Path
import socket

from .utils import read_json, sha256, write_json


def main():
    root=Path("part1_1_deepdish5_outputs")
    files=sorted(p for p in Path("part1_1").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    inputs=[Path("Assignment3-data")/name for name in ["README.txt","train_data.npy","test_data.npy"]]
    write_json(root/"source_manifest.json",dict(hostname=socket.gethostname(),
               source={str(p):sha256(p) for p in files},
               supplied_inputs={str(p):sha256(p) for p in inputs}))
    final=read_json(root/"final_test.json")
    lines=["# Part 1.1 completed artifacts", "", "All final experiments ran on DeepDish5.", "",
           f"**Final held-out accuracy: {100*final['accuracy']:.2f}% ({final['correct']:,}/{final['n']:,}).**", "",
           "- [Report notes with actual numerical results](REPORT_NOTES.md)",
           "- [Setup, exact reproduction commands and implementation guide](../part1_1/README.md)",
           "- [Six-setting grid plus four seed repeats](comparison.csv)",
           "- [Scientific validation: hypotheses, numerical results and limitations](validation/results.json)",
           "- [Correctness test log](validation/correctness_tests.log)",
           "- [Frozen validation-based selection](selection.json)",
           "- [Final test evaluation](final_test.json)",
           "- [Selected trained checkpoint](experiments/c10_e3_s2026/best.pt)",
           "- [Final artifact audit](final_audit.json)",
           "- [Hardware/environment](environment.json)",
           "- [Package versions](package_versions.txt)",
           "- [Source and supplied-input SHA-256 manifest](source_manifest.json)", "",
           "## Figures", "", "Each figure is saved as a 300-DPI PNG and vector PDF.", "",
           "| Figure | PNG | PDF |", "|---|---|---|"]
    for r in read_json(root/"figures/manifest.json"):
        name=r["name"]
        lines.append(f"| {name.replace('_',' ')} | [PNG](figures/{name}.png) | [PDF](figures/{name}.pdf) |")
    lines += ["", "The supplied references and source dataset are preserved. Earlier local pilots in `../part1_1_outputs/` are excluded from these results. Part 1.2 and Part 2 remain outside this implementation.", ""]
    (root/"INDEX.md").write_text("\n".join(lines),encoding="utf-8")
    print("Source/input manifest and artifact index written on",socket.gethostname())


if __name__ == "__main__":
    main()
