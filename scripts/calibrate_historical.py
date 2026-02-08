#!/usr/bin/env python3
"""
CLI script to run historical calibration and inspect results.

Usage:
    python3 scripts/calibrate_historical.py              # compute & cache
    python3 scripts/calibrate_historical.py --dry-run     # show diagnostics only (no cache write)
    python3 scripts/calibrate_historical.py --refresh      # force recompute
    python3 scripts/calibrate_historical.py --compare     # calibrated vs current hardcodes side-by-side
"""
from __future__ import annotations

import argparse
import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.historical_calibration import (
    HistoricalCalibration,
    compute_historical_calibration,
    load_historical_calibration,
    fallback_outcome_probs,
    fallback_linear_weights,
    fallback_steal_rates,
    fallback_gb_dp_rates,
    fallback_metric_ranges,
    fallback_fielding_benchmarks,
)
from config import PARQUET_PATH


def _section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_outcome_probs(cal: HistoricalCalibration) -> None:
    _section("1A. Batted Ball Outcome Probabilities")
    print(f"  {'Bucket':<10} {'xOut':>6} {'x1B':>6} {'x2B':>6} {'x3B':>6} {'xHR':>6} {'xErr':>6} {'N':>8}")
    print(f"  {'-'*10} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    for bucket in ["Barrel", "HiEV_FB", "Hard_LD", "GB", "Popup", "Soft", "Medium"]:
        d = cal.outcome_probs.get(bucket)
        if d is None:
            print(f"  {bucket:<10}   (no data)")
            continue
        total = d["xOut"] + d["x1B"] + d["x2B"] + d["x3B"] + d["xHR"] + d["xErr"]
        print(f"  {bucket:<10} {d['xOut']:>6.3f} {d['x1B']:>6.3f} {d['x2B']:>6.3f} "
              f"{d['x3B']:>6.3f} {d['xHR']:>6.3f} {d['xErr']:>6.3f} {d['n']:>8,}  "
              f"(sum={total:.3f})")


def print_linear_weights(cal: HistoricalCalibration) -> None:
    _section("1B. wOBA / Linear Weights")
    lw = cal.linear_weights
    for key in ["out_w", "bb_w", "hbp_w", "single_w", "double_w", "triple_w", "hr_w"]:
        print(f"  {key:<12} {lw.get(key, 0.0):>8.4f}")


def print_steal_rates(cal: HistoricalCalibration) -> None:
    _section("1C. Steal Success Rates by Pitch Velocity Class")
    sr = cal.steal_rates
    labels = {"slow_sb_pct": "<78 mph (slow)", "med_sb_pct": "78-84 mph (med)",
              "fast_sb_pct": "85-89 mph (fast)", "elite_sb_pct": "90+ mph (elite)"}
    for key, label in labels.items():
        print(f"  {label:<22} {sr.get(key, 0.0):>6.1f}%")


def print_gb_dp_rates(cal: HistoricalCalibration) -> None:
    _section("1D. GB% and DP% by Pitch Type")
    print(f"  {'Pitch':<15} {'GB%':>7} {'DP%':>7} {'Out%':>7} {'N':>8}")
    print(f"  {'-'*15} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
    for pt in ["Sinker", "Changeup", "Curveball", "Cutter", "Fastball", "Slider",
               "Splitter", "Knuckle Curve", "Sweeper"]:
        d = cal.gb_dp_rates.get(pt)
        if d is None:
            continue
        print(f"  {pt:<15} {d['gb_pct']:>7.1f} {d['dp_pct']:>7.2f} {d['out_pct']:>7.1f} {d['n_bip']:>8,}")
    avg_gb = cal.gb_dp_rates.get("_league_avg_gb", 47.0)
    avg_dp = cal.gb_dp_rates.get("_league_avg_dp", 4.1)
    print(f"\n  League avg GB%: {avg_gb:.1f}%    League avg DP%: {avg_dp:.2f}%")


def print_metric_ranges(cal: HistoricalCalibration) -> None:
    _section("1E. Pitch Metric Normalization Ranges (P5/P95)")
    mr = cal.metric_ranges
    print(f"  {'Metric':<12} {'P5':>8} {'P95':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8}")
    for metric in ["whiff", "csw", "chase"]:
        p5 = mr.get(f"{metric}_p5", float("nan"))
        p95 = mr.get(f"{metric}_p95", float("nan"))
        print(f"  {metric+'%':<12} {p5:>8.2f} {p95:>8.2f}")


def print_fielding_benchmarks(cal: HistoricalCalibration) -> None:
    _section("1F. D1 Fielding Benchmarks")
    fb = cal.fielding_benchmarks
    print(f"  FLD% median (from parquet):  {fb.get('fld_pct_median', 0):>8.4f}")
    print(f"  FLD% P75:                    {fb.get('fld_pct_p75', 0):>8.4f}")
    print(f"  FLD% P90:                    {fb.get('fld_pct_p90', 0):>8.4f}")
    print(f"  Error rate median:           {fb.get('error_rate_median', 0):>8.4f}")
    print(f"  N BIP:                       {fb.get('n_bip', 0):>8,}")


def print_comparison(cal: HistoricalCalibration) -> None:
    """Side-by-side: calibrated vs current hardcodes with deltas."""

    # ── Outcome probs comparison ──
    _section("COMPARE: Batted Ball Outcome Probabilities")
    fb = fallback_outcome_probs()
    print(f"  {'Bucket':<10} {'Metric':<8} {'Current':>8} {'Calibr':>8} {'Delta':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for bucket in ["Barrel", "HiEV_FB", "Hard_LD", "GB", "Popup", "Soft", "Medium"]:
        fb_d = fb.get(bucket, {})
        cal_d = cal.outcome_probs.get(bucket, {})
        for metric in ["xOut", "x1B", "x2B", "x3B", "xHR"]:
            curr = fb_d.get(metric, 0.0)
            calib = cal_d.get(metric, curr)
            delta = calib - curr
            flag = " ***" if abs(delta) >= 0.05 else ""
            print(f"  {bucket:<10} {metric:<8} {curr:>8.3f} {calib:>8.3f} {delta:>+8.3f}{flag}")

    # ── Linear weights comparison ──
    _section("COMPARE: wOBA / Linear Weights")
    fb_lw = fallback_linear_weights()
    print(f"  {'Weight':<12} {'Current':>8} {'Calibr':>8} {'Delta':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
    for key in ["out_w", "bb_w", "hbp_w", "single_w", "double_w", "triple_w", "hr_w"]:
        curr = fb_lw.get(key, 0.0)
        calib = cal.linear_weights.get(key, curr)
        delta = calib - curr
        print(f"  {key:<12} {curr:>8.4f} {calib:>8.4f} {delta:>+8.4f}")

    # ── Steal rates comparison ──
    _section("COMPARE: Steal Rates by Velocity Class")
    fb_sr = fallback_steal_rates()
    print(f"  {'Velo Class':<18} {'Current':>8} {'Calibr':>8} {'Delta':>8}")
    print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8}")
    for key in ["slow_sb_pct", "med_sb_pct", "fast_sb_pct", "elite_sb_pct"]:
        curr = fb_sr.get(key, 0.0)
        calib = cal.steal_rates.get(key, curr)
        delta = calib - curr
        print(f"  {key:<18} {curr:>8.1f} {calib:>8.1f} {delta:>+8.1f}")

    # ── Metric ranges comparison ──
    _section("COMPARE: Metric Normalization Ranges")
    fb_mr = fallback_metric_ranges()
    print(f"  {'Param':<14} {'Current':>8} {'Calibr':>8} {'Delta':>8}")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8}")
    for key in ["whiff_p5", "whiff_p95", "csw_p5", "csw_p95", "chase_p5", "chase_p95"]:
        curr = fb_mr.get(key, 0.0)
        calib = cal.metric_ranges.get(key, curr)
        delta = calib - curr
        print(f"  {key:<14} {curr:>8.2f} {calib:>8.2f} {delta:>+8.2f}")

    # ── Fielding benchmarks comparison ──
    _section("COMPARE: Fielding Benchmarks")
    fb_fb = fallback_fielding_benchmarks()
    print(f"  {'Param':<18} {'Current':>8} {'Calibr':>8} {'Delta':>8}")
    print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8}")
    for key in ["fld_pct_median", "fld_pct_p75", "fld_pct_p90", "error_rate_median"]:
        curr = fb_fb.get(key, 0.0)
        calib = cal.fielding_benchmarks.get(key, curr)
        delta = calib - curr
        print(f"  {key:<18} {curr:>8.4f} {calib:>8.4f} {delta:>+8.4f}")


def print_sanity_checks(cal: HistoricalCalibration) -> None:
    _section("Sanity Checks")
    ok = True

    # Outcome probs sum to ~1.0
    for bucket, d in cal.outcome_probs.items():
        total = d["xOut"] + d["x1B"] + d["x2B"] + d["x3B"] + d["xHR"] + d["xErr"]
        if abs(total - 1.0) > 0.02:
            print(f"  WARNING: {bucket} probs sum to {total:.3f} (expected ~1.0)")
            ok = False

    # Barrel HR% > other HR%
    barrel_hr = cal.outcome_probs.get("Barrel", {}).get("xHR", 0)
    for bucket, d in cal.outcome_probs.items():
        if bucket != "Barrel" and d.get("xHR", 0) > barrel_hr and barrel_hr > 0:
            print(f"  WARNING: {bucket} xHR={d['xHR']:.3f} > Barrel xHR={barrel_hr:.3f}")
            ok = False

    # GB out% should be high
    gb_out = cal.outcome_probs.get("GB", {}).get("xOut", 0)
    if gb_out < 0.50:
        print(f"  WARNING: GB xOut={gb_out:.3f} seems low (expected >0.50)")
        ok = False

    # Steal rates monotonically decrease with pitch velocity
    sr = cal.steal_rates
    if sr.get("slow_sb_pct", 100) < sr.get("elite_sb_pct", 0):
        print(f"  WARNING: Steal rates not monotonically decreasing with velo")
        ok = False

    # Linear weights monotonically increase: single < double < triple < hr
    lw = cal.linear_weights
    if not (lw["single_w"] <= lw["double_w"] <= lw["triple_w"] <= lw["hr_w"]):
        print(f"  WARNING: Linear weights not monotonically increasing")
        ok = False

    if ok:
        print("  All sanity checks passed.")


def main():
    parser = argparse.ArgumentParser(description="Historical calibration tool")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and display, but don't write cache")
    parser.add_argument("--refresh", action="store_true",
                        help="Force recompute even if cache exists")
    parser.add_argument("--compare", action="store_true",
                        help="Show calibrated vs current hardcodes side-by-side")
    parser.add_argument("--parquet", default=PARQUET_PATH,
                        help=f"Path to parquet file (default: {PARQUET_PATH})")
    args = parser.parse_args()

    if not os.path.exists(args.parquet):
        print(f"ERROR: Parquet file not found: {args.parquet}", file=sys.stderr)
        sys.exit(1)

    print(f"Parquet: {args.parquet}")
    print(f"Mode: {'dry-run' if args.dry_run else 'compute & cache'}")

    if args.dry_run:
        cal = compute_historical_calibration(parquet_path=args.parquet)
    else:
        cal = load_historical_calibration(parquet_path=args.parquet, force_refresh=args.refresh)
        if cal is None:
            print("ERROR: Could not compute calibration", file=sys.stderr)
            sys.exit(1)

    print_outcome_probs(cal)
    print_linear_weights(cal)
    print_steal_rates(cal)
    print_gb_dp_rates(cal)
    print_metric_ranges(cal)
    print_fielding_benchmarks(cal)
    print_sanity_checks(cal)

    if args.compare:
        print_comparison(cal)

    print("\nDone.")


if __name__ == "__main__":
    main()
