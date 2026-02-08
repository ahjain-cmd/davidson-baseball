#!/usr/bin/env python3
"""
CLI script to run count calibration and inspect results.

Usage:
    python3 scripts/calibrate_counts.py              # compute & cache
    python3 scripts/calibrate_counts.py --dry-run     # show diagnostics only (no cache write)
    python3 scripts/calibrate_counts.py --refresh      # force recompute
    python3 scripts/calibrate_counts.py --compare     # calibrated vs fallback weights side-by-side
"""
from __future__ import annotations

import argparse
import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.count_calibration import (
    CountCalibration,
    CountWeights,
    compute_count_calibration,
    fallback_weights,
    load_count_calibration,
)
from config import PARQUET_PATH


def _fmt(v: float, width: int = 7) -> str:
    return f"{v:>{width}.4f}"


def _fmt3(v: float, width: int = 7) -> str:
    return f"{v:>{width}.3f}"


def print_matrix(title: str, data: dict, fmt_fn=_fmt) -> None:
    """Print a 4×3 matrix (balls × strikes)."""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    header = "       " + "".join(f"  {s}K     " for s in range(3))
    print(header)
    for b in range(4):
        row = f"  {b}B   "
        for s in range(3):
            key = f"{b}-{s}"
            v = data.get(key)
            if v is not None:
                row += fmt_fn(v) + "  "
            else:
                row += "    -    "
        print(row)


def print_weights_table(cal: CountCalibration) -> None:
    """Print per-count calibrated weight table."""
    print(f"\n{'=' * 80}")
    print("  Per-Count Calibrated Weights")
    print(f"{'=' * 80}")
    print(f"  {'Count':<6} {'whiff_w':>8} {'csw_w':>8} {'chase_w':>8} "
          f"{'cmd_w':>8} {'hard_d':>8} {'os_d':>8}  {'PAs':>8}")
    print(f"  {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 8} "
          f"{'-' * 8} {'-' * 8} {'-' * 8}  {'-' * 8}")

    for b in range(4):
        for s in range(3):
            key = f"{b}-{s}"
            w = cal.weights.get(key)
            n = cal.pa_counts.get(key, 0)
            if w:
                print(f"  {key:<6} {w.whiff_w:>8.3f} {w.csw_w:>8.3f} {w.chase_w:>8.3f} "
                      f"{w.cmd_w:>8.3f} {w.hard_delta:>8.3f} {w.offspeed_delta:>8.3f}  {n:>8,}")


def print_comparison(cal: CountCalibration) -> None:
    """Side-by-side comparison of calibrated vs fallback weights."""
    fb = fallback_weights()
    print(f"\n{'=' * 100}")
    print("  Calibrated vs Fallback (Current Hardcoded) Weights — Delta shown")
    print(f"{'=' * 100}")
    fields = ["whiff_w", "csw_w", "chase_w", "cmd_w", "hard_delta", "offspeed_delta"]
    short = ["wh_w", "csw_w", "ch_w", "cmd_w", "hrd_d", "os_d"]

    header = f"  {'Count':<6}"
    for s in short:
        header += f" {'Cal':>5} {'FB':>5} {'Δ':>5} |"
    print(f"  {'':6}", end="")
    for s in short:
        print(f"  --- {s:^5} --- |", end="")
    print()
    print(header)
    print(f"  {'-' * 96}")

    for b in range(4):
        for s_val in range(3):
            key = f"{b}-{s_val}"
            cw = cal.weights.get(key)
            fw = fb.get(key)
            if not cw or not fw:
                continue
            line = f"  {key:<6}"
            for fld in fields:
                cv = getattr(cw, fld)
                fv = getattr(fw, fld)
                d = cv - fv
                line += f" {cv:>5.1f} {fv:>5.1f} {d:>+5.1f} |"
            print(line)


def main():
    parser = argparse.ArgumentParser(description="Count calibration tool")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and display, but don't write cache")
    parser.add_argument("--refresh", action="store_true",
                        help="Force recompute even if cache exists")
    parser.add_argument("--compare", action="store_true",
                        help="Show calibrated vs fallback weights side-by-side")
    parser.add_argument("--parquet", default=PARQUET_PATH,
                        help=f"Path to parquet file (default: {PARQUET_PATH})")
    args = parser.parse_args()

    if not os.path.exists(args.parquet):
        print(f"ERROR: Parquet file not found: {args.parquet}", file=sys.stderr)
        sys.exit(1)

    print(f"Parquet: {args.parquet}")
    print(f"Mode: {'dry-run' if args.dry_run else 'compute & cache'}")

    if args.dry_run:
        cal = compute_count_calibration(parquet_path=args.parquet)
    else:
        cal = load_count_calibration(parquet_path=args.parquet, force_refresh=args.refresh)
        if cal is None:
            print("ERROR: Could not compute calibration", file=sys.stderr)
            sys.exit(1)

    # Display results
    print_matrix("Run Values — RV(b,s)", cal.run_values)
    print_matrix("Strike Gain — RV(b,s) - RV(b,s+1)", cal.strike_gain)
    print_matrix("Ball Cost — RV(b+1,s) - RV(b,s)", cal.ball_cost)
    print_matrix("Chase→Whiff Rate", cal.chase_whiff_rate)
    print_weights_table(cal)

    # Sanity check display
    rv00 = cal.run_values.get("0-0", 0)
    print(f"\n  Sanity: RV(0-0) = {rv00:.4f}  (expected ~0.20-0.25)")

    if args.compare:
        print_comparison(cal)

    print("\nDone.")


if __name__ == "__main__":
    main()
