import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class RunRow:
    firma: str
    website: str
    irene_score: float
    model_score: float
    model: str


def _read_run_csv(path: Path) -> Dict[str, RunRow]:
    out: Dict[str, RunRow] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            website = (row.get("website") or row.get("Website") or "").strip()
            if not website:
                continue
            try:
                irene = float(row.get("irene_score") or row.get("Manuav-Score") or 0.0)
            except ValueError:
                continue
            try:
                model_score = float(row.get("model_score") or row.get("manuav_fit_score") or 0.0)
            except ValueError:
                continue
            out[website] = RunRow(
                firma=(row.get("firma") or row.get("Firma") or "").strip(),
                website=website,
                irene_score=irene,
                model_score=model_score,
                model=(row.get("model") or "").strip(),
            )
    return out


def _mae(pairs: List[Tuple[float, float]]) -> float:
    return sum(abs(a - b) for a, b in pairs) / len(pairs) if pairs else 0.0


def main() -> int:
    p = argparse.ArgumentParser(description="Compare multiple model runs against Irene for the 9-row sample.")
    p.add_argument("--run", action="append", required=True, help="CSV path for a run (repeatable).")
    args = p.parse_args()

    runs: List[Tuple[str, Dict[str, RunRow]]] = []
    for run_path in args.run:
        path = Path(run_path)
        data = _read_run_csv(path)
        label = path.stem
        runs.append((label, data))

    # Build website union
    websites = sorted({w for _label, d in runs for w in d.keys()})

    # Print header
    print("website, firma, irene_score, " + ", ".join([f"{label}_score, {label}_delta" for label, _d in runs]))

    # MAE per run
    maes: Dict[str, float] = {}
    for label, d in runs:
        pairs = [(row.irene_score, row.model_score) for row in d.values()]
        maes[label] = _mae(pairs)

    for w in websites:
        firma = ""
        irene: Optional[float] = None
        parts: List[str] = []
        for label, d in runs:
            rr = d.get(w)
            if rr is None:
                parts.extend(["", ""])
                continue
            if not firma:
                firma = rr.firma
            if irene is None:
                irene = rr.irene_score
            delta = rr.model_score - rr.irene_score
            parts.extend([f"{rr.model_score:.2f}", f"{delta:+.2f}"])

        if irene is None:
            continue
        print(f"{w}, {firma}, {irene:.2f}, " + ", ".join(parts))

    print("\nMAE vs Irene:")
    for label, _d in runs:
        print(f"- {label}: {maes[label]:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


