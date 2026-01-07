from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median


def _pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0


def _parse_int(v) -> int:
    try:
        return int(float(v))
    except Exception:
        return 0


def _parse_float(v) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze evaluate_list/run outputs (cost + web-search usage + timing).")
    ap.add_argument("--csv", required=True, help="Path to outputs CSV (e.g. outputs/<run>.csv)")
    args = ap.parse_args()

    p = Path(args.csv)
    rows = list(csv.DictReader(p.open("r", encoding="utf-8", newline="")))
    n = len(rows)
    if n == 0:
        print("No rows found.")
        return 2

    web_calls = [_parse_int(r.get("web_search_calls")) for r in rows]
    tool_total = [_parse_int(r.get("web_search_tool_calls_total")) for r in rows]
    q = [_parse_int(r.get("web_search_calls_query")) for r in rows]
    o = [_parse_int(r.get("web_search_calls_open")) for r in rows]
    u = [_parse_int(r.get("web_search_calls_unknown")) for r in rows]

    # Prefer query-count as the "billed" number if available; otherwise fall back to web_search_calls.
    billed = q if any(q) else web_calls

    cost = [_parse_float(r.get("cost_usd")) for r in rows]
    token_cost = [_parse_float(r.get("token_cost_usd")) for r in rows]
    ws_cost = [_parse_float(r.get("web_search_tool_cost_usd")) for r in rows]
    dur = [_parse_float(r.get("duration_seconds")) for r in rows]

    errors = sum(1 for r in rows if (r.get("error") or "").strip())

    print(f"rows: {n} (errors: {errors}, ok: {n - errors})")
    print(f"cost_usd: total={sum(cost):.4f} token={sum(token_cost):.4f} web_search_tool={sum(ws_cost):.4f}")
    print(f"duration: total_min={sum(dur)/60:.1f} mean_s={mean(dur):.2f} median_s={median(dur):.2f}")

    print("\nWeb search calls (billed estimate):")
    print(f"- total: {sum(billed)}  mean/row: {sum(billed)/n:.3f}")
    print(f"- distribution: {dict(sorted(Counter(billed).items()))}")

    if any(q):
        print("\nWeb search calls (from CSV column web_search_calls):")
        print(f"- total: {sum(web_calls)}  mean/row: {sum(web_calls)/n:.3f}")
        print(f"- distribution: {dict(sorted(Counter(web_calls).items()))}")

    if any(tool_total):
        print("\nWeb search tool calls (total tool invocations):")
        print(f"- total: {sum(tool_total)}  mean/row: {sum(tool_total)/n:.3f}")
        print(f"- distribution: {dict(sorted(Counter(tool_total).items()))}")

    if any(q) or any(o) or any(u):
        print("\nWeb search tool calls by kind (completed):")
        print(f"- query:   {sum(q)} (mean/row {sum(q)/n:.3f})")
        print(f"- open:    {sum(o)} (mean/row {sum(o)/n:.3f})")
        print(f"- unknown: {sum(u)} (mean/row {sum(u)/n:.3f})")
        if sum(q) + sum(o) + sum(u) > 0:
            tot = sum(q) + sum(o) + sum(u)
            print(f"- share: query={_pct(sum(q), tot):.1f}% open={_pct(sum(o), tot):.1f}% unknown={_pct(sum(u), tot):.1f}%")

    # Citations (usually empty when output is strict JSON without URLs)
    cites = []
    for r in rows:
        s = r.get("url_citations_json") or "[]"
        try:
            arr = json.loads(s)
            cites.append(len(arr) if isinstance(arr, list) else 0)
        except Exception:
            cites.append(0)
    print("\nCitations (url_citations_json):")
    print(f"- mean/row: {mean(cites):.2f}  median: {median(cites):.0f}  pct_zero: {_pct(sum(1 for x in cites if x==0), n):.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


