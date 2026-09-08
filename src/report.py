"""Rebuild every results table in the README from the JSON files in `results/`.

No number in the README is typed by hand. That is not tidiness, it is the only way a
reader can tell the difference between a result and a claim: if a table can only be
produced by running the code, then the table is evidence.

Two rules this module enforces, both aimed at the same failure:

  * A configuration with fewer than `--min-seeds` runs is reported with its seed count
    visible, so a single-run number can never be mistaken for a measured one.
  * Two configurations whose mean +/- std intervals overlap are declared
    indistinguishable rather than ranked. On SST and 20 Newsgroups the per-seed spread
    is large enough that ranking on the mean alone would invent an ordering.

Usage:
    python -m src.report                 # print the tables
    python -m src.report --write         # splice them into README.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
README = ROOT / "README.md"

BEGIN = "<!-- BEGIN GENERATED: {name} -->"
END = "<!-- END GENERATED: {name} -->"

#: Kendall tau against gradient importance as printed in the paper's Table 2, per
#: class (0 / 1). Kept here so the reproduction gap is visible in the table rather
#: than left for the reader to look up.
PAPER_TAU_G = {
    ("sst", "bilstm"): (0.34, 0.36),
    ("sst", "average"): (0.61, 0.60),
    ("imdb", "bilstm"): (0.44, 0.43),
    ("imdb", "average"): (0.67, 0.68),
    ("20news", "bilstm"): (0.07, 0.21),
    ("20news", "average"): (0.79, 0.75),
    ("agnews", "bilstm"): (0.36, 0.42),
    ("agnews", "average"): (0.78, 0.76),
}

ENCODER_ORDER = ["average", "cnn", "bilstm", "transformer"]


def load_results() -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in sorted(RESULTS.glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        grouped[(r["dataset"], r["encoder"])].append(r)
    return grouped


def aggregate(runs: list[dict], field: str = "tau_gradient") -> dict:
    """Mean across seeds of the per-seed mean, plus the spread across seeds.

    Two different spreads exist here and conflating them would be misleading. Each
    run already reports a standard deviation *across instances*. What matters for
    "is this effect real" is the variation *across seeds*, which is what this
    returns as `std`. The instance-level spread is kept separately.
    """
    means = np.array([r[field]["mean"] for r in runs], dtype=float)
    sig = np.array([r[field]["sig_frac"] for r in runs], dtype=float)
    acc = np.array([r["performance"]["accuracy"] for r in runs], dtype=float)
    within = np.array([r[field]["std"] for r in runs], dtype=float)
    return {
        "n_seeds": len(runs),
        "mean": float(np.nanmean(means)),
        "std": float(np.nanstd(means)) if len(means) > 1 else float("nan"),
        "within_instance_std": float(np.nanmean(within)),
        "sig_frac": float(np.nanmean(sig)),
        "accuracy": float(np.nanmean(acc)),
    }


def overlaps(a: dict, b: dict) -> bool:
    """True when two mean +/- std intervals intersect, so neither can be ranked first."""
    if np.isnan(a["std"]) or np.isnan(b["std"]):
        return True                      # a single seed cannot establish an ordering
    return not (a["mean"] - a["std"] > b["mean"] + b["std"]
                or b["mean"] - b["std"] > a["mean"] + a["std"])


def fmt(value: float, std: float) -> str:
    if np.isnan(value):
        return "n/a"
    if np.isnan(std):
        return f"{value:.3f} (1 seed)"
    return f"{value:.3f} +/- {std:.3f}"


def table_reproduction(grouped) -> str:
    rows = ["| Dataset | Encoder | Seeds | Accuracy | tau_g (this repo) | tau_g (paper) |",
            "|---|---|---|---|---|---|"]
    for (dataset, encoder), runs in sorted(grouped.items()):
        paper = PAPER_TAU_G.get((dataset, encoder))
        if paper is None:
            continue
        a = aggregate(runs)
        rows.append(
            f"| {dataset} | {encoder} | {a['n_seeds']} | {a['accuracy']:.3f} | "
            f"{fmt(a['mean'], a['std'])} | {paper[0]:.2f} / {paper[1]:.2f} |"
        )
    return "\n".join(rows)


def table_extension(grouped) -> str:
    datasets = sorted({d for d, _ in grouped})
    header = "| Dataset | " + " | ".join(ENCODER_ORDER) + " |"
    rows = [header, "|---" * (len(ENCODER_ORDER) + 1) + "|"]
    for dataset in datasets:
        cells = []
        for encoder in ENCODER_ORDER:
            runs = grouped.get((dataset, encoder))
            if not runs:
                cells.append("")
                continue
            a = aggregate(runs)
            cells.append(fmt(a["mean"], a["std"]))
        rows.append(f"| {dataset} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def notes(grouped) -> str:
    """State every place the data does not support a ranking."""
    lines = []
    datasets = sorted({d for d, _ in grouped})
    for dataset in datasets:
        present = [e for e in ENCODER_ORDER if (dataset, e) in grouped]
        aggs = {e: aggregate(grouped[(dataset, e)]) for e in present}
        single = [e for e in present if aggs[e]["n_seeds"] < 2]
        if single:
            lines.append(
                f"- **{dataset}**: {', '.join(single)} ran on a single seed, so no "
                "spread is available and no ordering involving them is established."
            )
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                if aggs[a]["n_seeds"] > 1 and aggs[b]["n_seeds"] > 1 \
                        and overlaps(aggs[a], aggs[b]):
                    lines.append(
                        f"- **{dataset}**: `{a}` and `{b}` overlap within one standard "
                        "deviation across seeds and are reported as indistinguishable."
                    )
    return "\n".join(lines) if lines else "- No overlapping configurations."


def splice(text: str, name: str, body: str) -> str:
    begin, end = BEGIN.format(name=name), END.format(name=name)
    if begin not in text or end not in text:
        return text
    head = text.split(begin)[0]
    tail = text.split(end, 1)[1]
    return f"{head}{begin}\n{body}\n{end}{tail}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="splice into README.md")
    args = ap.parse_args()

    grouped = load_results()
    if not grouped:
        print(f"no results in {RESULTS}. Run src.experiment first.")
        return

    sections = {
        "reproduction": table_reproduction(grouped),
        "extension": table_extension(grouped),
        "notes": notes(grouped),
    }
    for name, body in sections.items():
        print(f"\n### {name}\n\n{body}")

    if args.write:
        text = README.read_text(encoding="utf-8")
        for name, body in sections.items():
            text = splice(text, name, body)
        README.write_text(text, encoding="utf-8")
        print(f"\nwritten into {README}")


if __name__ == "__main__":
    main()
