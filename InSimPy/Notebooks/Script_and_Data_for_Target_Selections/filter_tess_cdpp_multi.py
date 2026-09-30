#!/usr/bin/env python3
"""
Filter TIC IDs from one or many TESS RMS‑CDPP CSVs, with support for:

  • Single‑sector files → merged together → one output CSV + one output LOG
  • Multi‑sector files → each processed separately → one CSV + one LOG per file
  • Three filtering modes:
        --cdpp-thresh X
        --cdpp-percentile P
        --cdpp-percentile-range P1 P2
  • Percentile scope: window | global
  • User‑chosen output directory: --outdir DIR
  • Human-readable or automatic timestamp tagging for filenames:
        --tag TAG
        (If omitted: use YYYYMMDDThhmmssZ timestamp)
  • Minimal CSV outputs:
        single-sector: ticid,sector
        multi-sector:  ticid,sector_range
  • Log files contain summary (not full TIC lists)

Author: Generated for your workflow
"""

import argparse
import os
import re
import pandas as pd
import numpy as np
from datetime import datetime


# --------------------------------------------------------------------------
# TAG handling
# --------------------------------------------------------------------------

def make_tag(user_tag: str | None) -> str:
    """Return user tag if provided, else a UTC timestamp tag."""
    if user_tag:
        return user_tag
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


# --------------------------------------------------------------------------
# Sector parsing
# --------------------------------------------------------------------------

def extract_sector_info(path: str):
    """
    Reads comment header lines from CSV and returns:
        ("single", sector_number)
        ("range",  "X-Y")
        ("unknown", None)
    """
    # Patterns
    single_pat = re.compile(r"Sector\s*:\s*(\d+)", re.IGNORECASE)

    range_pats = [
        re.compile(r"Sector[s]?\s*:\s*(\d+)\s*[-–—]\s*(\d+)", re.IGNORECASE),
        re.compile(r"Sector[s]?\s*\(\s*\)\s*:\s*(\d+)\s*[-–—]\s*(\d+)", re.IGNORECASE),
        re.compile(r"Sector[s]?\s*:\s*(\d+)\s*to\s*(\d+)", re.IGNORECASE),
    ]

    try:
        with open(path, "r") as f:
            for line in f:
                if not line.startswith("#"):
                    break

                for pat in range_pats:
                    m = pat.search(line)
                    if m:
                        lo, hi = m.group(1), m.group(2)
                        return ("range", f"{lo}-{hi}")

                m = single_pat.search(line)
                if m:
                    return ("single", int(m.group(1)))

    except Exception:
        return ("unknown", None)

    return ("unknown", None)


# --------------------------------------------------------------------------
# CDPP filtering helpers
# --------------------------------------------------------------------------

def load_cdpp_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip().lower() for c in df.columns]
    if "ticid" not in df.columns or "tmag" not in df.columns:
        raise ValueError(f"CSV {path} missing 'ticid' or 'tmag' columns.")
    return df


def validate_cdpp_col(df: pd.DataFrame, col: str) -> str:
    col = col.strip().lower()
    if col not in df.columns:
        cands = [c for c in df.columns if "cdpp" in c]
        raise ValueError(f"CDPP column '{col}' not found. Try one of: {cands}")
    return col


def filter_by_tmag(df: pd.DataFrame, tmin: float, tmax: float) -> pd.DataFrame:
    mask = np.isfinite(df["tmag"]) & (df["tmag"] >= tmin) & (df["tmag"] <= tmax)
    return df.loc[mask].copy()


def apply_threshold(df_win: pd.DataFrame, col: str, thresh: float) -> pd.DataFrame:
    d = df_win[np.isfinite(df_win[col])]
    return d.loc[d[col] < thresh].copy()


def apply_single_percentile(df_win: pd.DataFrame, df_all: pd.DataFrame,
                             col: str, pct: float, scope: str):
    src = df_win if scope == "window" else df_all
    vals = src[col].to_numpy()
    vals = vals[np.isfinite(vals)]
    cutoff = np.percentile(vals, pct)
    sel = df_win[(np.isfinite(df_win[col])) & (df_win[col] <= cutoff)].copy()
    return sel, cutoff


def apply_percentile_range(df_win: pd.DataFrame, df_all: pd.DataFrame,
                           col: str, pct_low: float, pct_high: float, scope: str):
    src = df_win if scope == "window" else df_all
    vals = src[col].to_numpy()
    vals = vals[np.isfinite(vals)]
    lowv = np.percentile(vals, pct_low)
    highv = np.percentile(vals, pct_high)
    sel = df_win[(np.isfinite(df_win[col])) &
                 (df_win[col] >= lowv) &
                 (df_win[col] <= highv)].copy()
    return sel, lowv, highv


# --------------------------------------------------------------------------
# Logging helper
# --------------------------------------------------------------------------

def append_log(path: str, text: str):
    """Append text to output log file."""
    with open(path, "a") as f:
        f.write(text + "\n")


# --------------------------------------------------------------------------
# Process a single CSV file
# --------------------------------------------------------------------------

def process_csv_file(path: str, args, tag: str,
                     combined_single_rows: list,
                     combined_single_log_entries: list):
    print(f"\n=== Processing file: {path} ===")

    # Determine sector type
    stype, sval = extract_sector_info(path)
    print(f"  Sector info: {stype} ({sval})")

    df = load_cdpp_csv(path)
    cdpp_col = validate_cdpp_col(df, args.cdpp_col)

    # Tmag range
    if args.tmag_center is not None:
        tmin = args.tmag_center - args.tmag_halfwidth
        tmax = args.tmag_center + args.tmag_halfwidth
    else:
        tmin, tmax = args.tmag_range

    df_win = filter_by_tmag(df, tmin, tmax)

    # CDPP mode
    summary_details = []

    if args.cdpp_thresh is not None:
        sel = apply_threshold(df_win, cdpp_col, args.cdpp_thresh)
        summary_details.append(f"CDPP MODE: threshold < {args.cdpp_thresh} ppm")
        summary_details.append(f"ROWS SELECTED: {len(sel)}")

    elif args.cdpp_percentile is not None:
        pct = args.cdpp_percentile
        sel, cutoff = apply_single_percentile(df_win, df, cdpp_col, pct, args.percentile_scope)
        summary_details.append(f"CDPP MODE: percentile <= {pct}%")
        summary_details.append(f"CUTOFF: {cutoff:.3f} ppm")
        summary_details.append(f"ROWS SELECTED: {len(sel)}")

    else:
        pct_low, pct_high = args.cdpp_percentile_range
        sel, lowv, highv = apply_percentile_range(df_win, df, cdpp_col,
                                                  pct_low, pct_high,
                                                  args.percentile_scope)
        summary_details.append(f"CDPP MODE: percentile range {pct_low}%–{pct_high}%")
        summary_details.append(f"CUTOFF RANGE: {lowv:.3f}–{highv:.3f} ppm")
        summary_details.append(f"ROWS SELECTED: {len(sel)}")

    # Output handling
    if stype == "single":
        for tic in sel["ticid"]:
            combined_single_rows.append((int(tic), sval))

        # Add log entry for this file
        block = []
        block.append("=" * 60)
        block.append(f"INPUT CSV: {path}")
        block.append(f"SECTOR TYPE: single")
        block.append(f"SECTOR: {sval}")
        block.append(f"TMAG RANGE: {tmin:.3f}–{tmax:.3f}")
        block.append(f"CDPP COLUMN: {cdpp_col}")
        block.extend(summary_details)
        combined_single_log_entries.append("\n".join(block))

    elif stype == "range":
        # Write separate CSV + log
        out_csv = os.path.join(args.outdir, f"filtered_multi_sector_{sval}_{tag}.csv")
        out_log = os.path.join(args.outdir, f"filtered_multi_sector_{sval}_{tag}.log")

        df_out = pd.DataFrame({
            "ticid": sel["ticid"].astype(int),
            "sector_range": sval
        })
        df_out.to_csv(out_csv, index=False)
        print(f"  Wrote {len(df_out)} rows → {out_csv}")

        # Log file
        block = []
        block.append("=" * 60)
        block.append(f"INPUT CSV: {path}")
        block.append(f"SECTOR TYPE: multi-sector")
        block.append(f"SECTOR RANGE: {sval}")
        block.append(f"TMAG RANGE: {tmin:.3f}–{tmax:.3f}")
        block.append(f"CDPP COLUMN: {cdpp_col}")
        block.extend(summary_details)
        block.append(f"OUTPUT CSV: {os.path.basename(out_csv)}")

        append_log(out_log, "\n".join(block))
        print(f"  Wrote log → {out_log}")

    else:
        print("  WARNING: Sector not detected → file ignored.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Filter TIC IDs from multiple TESS CDPP CSV files.")
    ap.add_argument("csv_files", nargs="*", help="Individual CSV files.")
    ap.add_argument("--folder", type=str, default=None, help="Folder containing CSVs.")

    # Output folder
    ap.add_argument("--outdir", type=str, default=".", help="Output directory.")

    # Tagging
    ap.add_argument("--tag", type=str, default=None,
                    help="Optional tag for output filenames. If omitted, timestamp is used.")

    # Tmag window
    grp_mag = ap.add_mutually_exclusive_group(required=True)
    grp_mag.add_argument("--tmag-center", type=float)
    ap.add_argument("--tmag-halfwidth", type=float)
    grp_mag.add_argument("--tmag-range", nargs=2, type=float)

    # CDPP modes
    grp_cdpp = ap.add_mutually_exclusive_group(required=True)
    grp_cdpp.add_argument("--cdpp-thresh", type=float)
    grp_cdpp.add_argument("--cdpp-percentile", type=float)
    grp_cdpp.add_argument("--cdpp-percentile-range", nargs=2, type=float,
                           metavar=("LOW", "HIGH"))

    ap.add_argument("--percentile-scope", choices=["window", "global"], default="window")
    ap.add_argument("--cdpp-col", default="rrmscdpp01p0")

    args = ap.parse_args()

    # Ensure output directory exists
    os.makedirs(args.outdir, exist_ok=True)

    # Determine final run TAG
    tag = make_tag(args.tag)

    # Collect CSV paths
    all_csvs = []

    for f in args.csv_files:
        if f.lower().endswith(".csv"):
            all_csvs.append(f)

    if args.folder:
        for fn in os.listdir(args.folder):
            if fn.lower().endswith(".csv"):
                all_csvs.append(os.path.join(args.folder, fn))

    if not all_csvs:
        print("No CSV files found.")
        return

    print("Found CSV files:")
    for f in all_csvs:
        print("  ", f)

    # Accumulators for single-sector combined results
    combined_single_rows = []
    combined_single_log_entries = []

    # Process each file
    for path in all_csvs:
        process_csv_file(path, args, tag, combined_single_rows, combined_single_log_entries)

    # Write combined single-sector outputs
    if combined_single_rows:
        out_csv = os.path.join(args.outdir, f"filtered_single_sectors_ticids_{tag}.csv")
        out_log = os.path.join(args.outdir, f"filtered_single_sectors_ticids_{tag}.log")

        df_out = pd.DataFrame(combined_single_rows, columns=["ticid", "sector"])
        df_out = df_out.drop_duplicates().sort_values(["sector", "ticid"])
        df_out.to_csv(out_csv, index=False)

        print(f"\nWrote combined single-sector CSV → {out_csv} ({len(df_out)} rows)")

        # Write combined log
        with open(out_log, "w") as f:
            for block in combined_single_log_entries:
                f.write(block + "\n\n")

        print(f"Wrote combined single-sector log → {out_log}")

    else:
        print("\nNo single-sector selections produced.")


if __name__ == "__main__":
    main()
