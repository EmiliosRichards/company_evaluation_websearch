from __future__ import annotations

from manuav_eval.costing import PricingPer1M, compute_cost_usd


class _Usage:
    input_tokens = 1_000_000
    output_tokens = 1_000_000
    total_tokens = 2_000_000
    input_tokens_details = type("X", (), {"cached_tokens": 100_000})()
    output_tokens_details = type("Y", (), {"reasoning_tokens": 0})()


def test_compute_cost_usd_accounts_for_cached_tokens() -> None:
    usage = _Usage()
    pricing = PricingPer1M(input_usd=1.75, cached_input_usd=0.175, output_usd=14.0)

    # non_cached = 900k => 0.9*1.75 = 1.575
    # cached = 100k => 0.1*0.175 = 0.0175
    # output = 1.0*14 = 14.0
    # total = 15.5925
    cost = compute_cost_usd(usage, pricing)
    assert abs(cost - 15.5925) < 1e-9


