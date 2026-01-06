import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class RowPick:
    bucket: str  # low/mid/high
    firma: str
    website: str
    score: float
    kurzurteil: str


def _parse_score(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _bucket_fixed(score: float) -> str | None:
    # Matches Manuav score bands.
    if score <= 3:
        return "low"
    if 4 <= score <= 6:
        return "mid"
    if score >= 7:
        return "high"
    # (3,4) and (6,7) gaps shouldn't happen if scores are integers; treat as mid.
    return "mid"


def _bucket_tertiles(score: float, p33: float, p67: float) -> str:
    if score <= p33:
        return "low"
    if score >= p67:
        return "high"
    return "mid"


def _compute_percentiles(values: List[float]) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    xs = sorted(values)

    def pct(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = (len(xs) - 1) * p
        lo = int(idx)
        hi = min(lo + 1, len(xs) - 1)
        frac = idx - lo
        return xs[lo] * (1 - frac) + xs[hi] * frac

    return pct(0.33), pct(0.67)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a 9-row sample from Irene's CSV (3 low/3 mid/3 high).")
    parser.add_argument(
        "--input",
        default="data/Websearch Irene - Manuav AI Search.csv",
        help="Path to Irene CSV",
    )
    parser.add_argument(
        "--output",
        default="data/irene_sample_9.csv",
        help="Where to write the 9-row sample CSV",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic sampling seed",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"Firma", "Website", "Manuav-Score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required columns: {sorted(missing)}. Found: {reader.fieldnames}")

        rows: List[Dict[str, str]] = []
        scores: List[float] = []
        for r in reader:
            score = _parse_score(r.get("Manuav-Score", ""))
            website = (r.get("Website") or "").strip()
            if score is None or not website:
                continue
            rows.append(r)
            scores.append(score)

    if not rows:
        raise SystemExit("No rows with both Website and Manuav-Score found.")

    # First attempt: fixed Manuav bands.
    buckets: Dict[str, List[Dict[str, str]]] = {"low": [], "mid": [], "high": []}
    for r in rows:
        s = _parse_score(r.get("Manuav-Score", ""))  # exists
        b = _bucket_fixed(float(s))
        if b in buckets:
            buckets[b].append(r)

    use_tertiles = any(len(buckets[b]) < 3 for b in ("low", "mid", "high"))
    p33, p67 = _compute_percentiles(scores)
    if use_tertiles:
        buckets = {"low": [], "mid": [], "high": []}
        for r in rows:
            s = float(_parse_score(r.get("Manuav-Score", "")))
            b = _bucket_tertiles(s, p33, p67)
            buckets[b].append(r)

    rng = random.Random(args.seed)
    picks: List[RowPick] = []
    for b in ("low", "mid", "high"):
        if len(buckets[b]) < 3:
            raise SystemExit(f"Not enough rows in bucket '{b}' to sample 3 (found {len(buckets[b])}).")
        chosen = rng.sample(buckets[b], 3)
        for r in chosen:
            score = float(_parse_score(r.get("Manuav-Score", "")))
            picks.append(
                RowPick(
                    bucket=b,
                    firma=(r.get("Firma") or "").strip(),
                    website=(r.get("Website") or "").strip(),
                    score=score,
                    kurzurteil=(r.get("Kurzurteil") or "").strip(),
                )
            )

    picks.sort(key=lambda x: ({"low": 0, "mid": 1, "high": 2}[x.bucket], x.score, x.firma.lower()))

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["bucket", "Firma", "Website", "Manuav-Score", "Kurzurteil"],
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for p in picks:
            writer.writerow(
                {
                    "bucket": p.bucket,
                    "Firma": p.firma,
                    "Website": p.website,
                    "Manuav-Score": f"{p.score:g}",
                    "Kurzurteil": p.kurzurteil,
                }
            )

    note = "fixed bands" if not use_tertiles else f"tertiles (p33={p33:g}, p67={p67:g})"
    print(f"Wrote {output_path} with 9 rows using {note}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


