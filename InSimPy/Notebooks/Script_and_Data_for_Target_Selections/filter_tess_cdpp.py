#!/usr/bin/env python3
"""
Filter TIC IDs by TESS magnitude and CDPP (threshold or percentile).

New features:
  * Automatically extracts SECTOR from header comments (e.g., "## Sector: 90").
  * CSV output contains ONLY: ticid,sector.
  * Console output still prints diagnostic values (tmag/CDPP).

Supports:
  --cdpp-thresh       (absolute threshold in ppm)
  --cdpp-percentile   (best X% by CDPP, within window or global)

Default CDPP column: rrmscdpp01p0  (Robust RMS CDPP at 1.0 hr)
"""

import argparse
import pandas as pd
import numpy as np
import re


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def extract_sector(path: str) -> int | None:
    """Parse the CSV header comments to find a 'Sector: N' entry."""
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"):
                m = re.search(r"Sector:\s*(\d+)", line)
                if m:
                    return int(m.group(1))
            else:
                break
    return None


def load_cdpp_csv(path: str) -> pd.DataFrame:
    """Load CSV and normalize column names."""
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip().lower() for c in df.columns]
    if "ticid" not in df.columns or "tmag" not in df.columns:
        raise ValueError("CSV missing required columns 'ticid' and/or 'tmag'.")
    return df


def validate_cdpp_column(df: pd.DataFrame, cdpp_col: str) -> str:
    """Ensure chosen CDPP column exists."""
    cdpp_col = cdpp_col.strip().lower()
    if cdpp_col not in df.columns:
        candidates = [c for c in df.columns if "cdpp" in c]
        raise ValueError(
            f"CDPP column '{cdpp_col}' not found. "
            f"Possible CDPP columns:\n  {', '.join(candidates)}"
        )
    return cdpp_col


def filter_by_tmag(df: pd.DataFrame, tmin: float, tmax: float) -> pd.DataFrame:
    """Subset dataframe by Tmag window."""
    mask = np.isfinite(df["tmag"]) & (df["tmag"] >= tmin) & (df["tmag"] <= tmax)
    return df.loc[mask].copy()


def select_by_threshold(df_win: pd.DataFrame, cdpp_col: str, threshold_ppm: float) -> pd.DataFrame:
    """Select rows with CDPP < threshold."""
    d = df_win[np.isfinite(df_win[cdpp_col])]
    return d.loc[d[cdpp_col] < threshold_ppm].copy()


def select_by_percentile(
    df_window: pd.DataFrame,
    df_all: pd.DataFrame,
    cdpp_col: str,
    percentile: float,
    scope: str = "window",
) -> tuple[pd.DataFrame, float]:
    """Select TICs with CDPP below given percentile."""
    if not (0 < percentile < 100):
        raise ValueError("--cdpp-percentile must be in (0,100).")

    src = df_window if scope == "window" else df_all
    vals = src[cdpp_col].to_numpy()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("No CDPP values available to compute percentile.")

    cutoff = np.percentile(vals, percentile)
    sel = df_window[np.isfinite(df_window[cdpp_col]) & (df_window[cdpp_col] <= cutoff)].copy()
    return sel, float(cutoff)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Select TIC IDs by TESS magnitude and CDPP (threshold or percentile)."
    )
    ap.add_argument("csv", help="Path to TESS rms‑cdpp CSV (e.g., tess2025...rms-cdpp.csv)")

    # Tmag range
    grp_mag = ap.add_mutually_exclusive_group(required=True)
    grp_mag.add_argument("--tmag-center", type=float)
    ap.add_argument("--tmag-halfwidth", type=float)
    grp_mag.add_argument("--tmag-range", nargs=2, type=float, metavar=("TMIN", "TMAX"))

    # CDPP selection
    grp_sel = ap.add_mutually_exclusive_group(required=True)
    grp_sel.add_argument("--cdpp-thresh", type=float)
    grp_sel.add_argument("--cdpp-percentile", type=float)

    ap.add_argument("--percentile-scope", choices=["window", "global"], default="window")
    ap.add_argument("--cdpp-col", default="rrmscdpp01p0")
    ap.add_argument("--out", type=str, default=None,
                    help="If set, CSV output will contain only: ticid,sector")

    args = ap.parse_args()

    # Tmag window
    if args.tmag_center is not None:
        if args.tmag_halfwidth is None:
            ap.error("--tmag-center requires --tmag-halfwidth")
        tmin = args.tmag_center - args.tmag_halfwidth
        tmax = args.tmag_center + args.tmag_halfwidth
    else:
        tmin, tmax = args.tmag_range

    # Load CSV + detect sector
    df = load_cdpp_csv(args.csv)
    sector = extract_sector(args.csv)  # may be None if not found
    if sector is None:
        sector="Multisector"

    cdpp_col = validate_cdpp_column(df, args.cdpp_col)

    # Filter by tmag window
    df_win = filter_by_tmag(df, tmin, tmax)

    # CDPP filtering mode
    if args.cdpp_thresh is not None:
        selected = select_by_threshold(df_win, cdpp_col, args.cdpp_thresh)
        info = f"threshold = {args.cdpp_thresh:.3f} ppm"
    else:
        selected, cutoff = select_by_percentile(
            df_window=df_win,
            df_all=df,
            cdpp_col=cdpp_col,
            percentile=args.cdpp_percentile,
            scope=args.percentile_scope,
        )
        info = f"percentile = {args.cdpp_percentile:.2f}% ; cutoff = {cutoff:.3f} ppm"

    # Add sector column (constant)
    selected = selected.sort_values(["tmag", cdpp_col, "ticid"]).reset_index(drop=True)
    selected["sector"] = sector

    # Console print (diagnostic)
    print(f"# Sector: {sector}")
    print(f"# Tmag in [{tmin:.3f}, {tmax:.3f}]")
    print(f"# CDPP column: {cdpp_col}")
    print(f"# Selection: {info}")
    print(f"# Rows selected: {len(selected)}\n")

    for _, r in selected.iterrows():
        print(f"TIC {int(r['ticid'])}\tTmag={r['tmag']:.3f}\tCDPP={r[cdpp_col]:.3f}\tSector={sector}")

    # CSV output — minimal columns
    if args.out:
        selected[["ticid", "sector"]].to_csv(args.out, index=False)
        print(f"\nWrote {len(selected)} rows to {args.out} (columns: ticid, sector)")


if __name__ == "__main__":
    main()
