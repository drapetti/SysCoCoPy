#!/usr/bin/env python3
"""
Plot CDPP vs TESS magnitude from TESS RMS-CDPP catalog CSV.

Features:
  * Auto-detection of Tmag column (robust)
  * Optional manual Tmag override:        --tmag-col TMAG_COLUMN_NAME
  * Selectable CDPP column:               --cdpp-col rrmscdpp01p0
  * Tmag filtering via --tmag-range
  * Optional clipping of extreme CDPP values
  * Hexbin or scatter plot
  * Log-scale option
  * Median CDPP per magnitude bin
  * Optional percentile envelope curves via:
        --percentiles LOW HIGH
    Example:
        --percentiles 5 95 --bin-width 0.1
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unicodedata


# ---------------------------------------------------------
# Utility: robust column normalization
# ---------------------------------------------------------

def normalize_column(col):
    """Lowercase, strip whitespace, normalize unicode, replace spaces."""
    col = unicodedata.normalize("NFKC", col)
    col = col.strip().lower()
    col = col.replace(" ", "_")
    col = col.replace("\t", "")
    return col


def autodetect_tmag_column(cols):
    """
    Auto-detect the Tmag column from normalized column names.
    Priority order:
        1. exact 'tmag'
        2. exact 't_mag'
        3. any column containing 'tmag'
    """
    if "tmag" in cols:
        return "tmag"
    if "t_mag" in cols:
        return "t_mag"

    # fallback: any col containing substring
    candidates = [c for c in cols if "tmag" in c]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # choose shortest name
        return sorted(candidates, key=len)[0]
    return None


# ---------------------------------------------------------
# CDPP label expansion
# ---------------------------------------------------------

def _duration_token_to_hours(token: str):
    try:
        h, m = token.split("p")
        return float(f"{int(h)}.{int(m)}")
    except Exception:
        return None


def expand_cdpp_label(colname: str) -> str:
    name = colname.strip().lower()
    if name.startswith("rrmscdpp"):
        token = name.replace("rrmscdpp", "")
        hr = _duration_token_to_hours(token)
        if hr is not None:
            return f"Robust RMS CDPP at {hr:.1f} hr"
        return "Robust RMS CDPP"
    if name.startswith("rmscdpp"):
        token = name.replace("rmscdpp", "")
        hr = _duration_token_to_hours(token)
        if hr is not None:
            return f"RMS CDPP at {hr:.1f} hr"
        return "RMS CDPP"
    if "cdpp" in name:
        return "CDPP"
    return colname


# ---------------------------------------------------------
# Bin statistics (median + percentiles)
# ---------------------------------------------------------

def compute_bin_statistics(df, bin_width, cdpp_col, tmag_col, percentiles=None):
    """
    Compute per-Tmag-bin statistics: median and optional percentile bands.
    """
    tvals = df[tmag_col].values
    cvals = df[cdpp_col].values

    tmin, tmax = np.min(tvals), np.max(tvals)
    edges = np.arange(
        np.floor(tmin / bin_width) * bin_width,
        np.ceil(tmax / bin_width) * bin_width + bin_width,
        bin_width
    )

    idx = np.digitize(tvals, edges) - 1
    rows = []

    for i in range(len(edges) - 1):
        sel = idx == i
        if not np.any(sel):
            continue

        t_center = 0.5 * (edges[i] + edges[i+1])
        vals = cvals[sel]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue

        out = dict()
        out["tmag_center"] = t_center
        out["median"] = np.median(vals)

        if percentiles:
            p_low, p_high = percentiles
            out["p_low"] = np.percentile(vals, p_low)
            out["p_high"] = np.percentile(vals, p_high)

        rows.append(out)

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------

def make_plot(df, tmag_col, cdpp_col, logy, hexbin, gridsize,
              bin_width, percentiles, title, outpath):
    label_text = expand_cdpp_label(cdpp_col)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Scatter or hexbin
    if hexbin:
        hb = ax.hexbin(
            df[tmag_col].values,
            df[cdpp_col].values,
            gridsize=gridsize,
            mincnt=1,
            cmap="viridis",
            bins="log" if logy else None
        )
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Counts per bin")
    else:
        ax.scatter(
            df[tmag_col].values,
            df[cdpp_col].values,
            s=6, alpha=0.6, linewidths=0, c="tab:blue", rasterized=True
        )

    # Median + percentiles
    if bin_width:
        stats = compute_bin_statistics(
            df, bin_width=bin_width, cdpp_col=cdpp_col,
            tmag_col=tmag_col, percentiles=percentiles
        )

        # median
        ax.plot(
            stats["tmag_center"], stats["median"],
            color="red", lw=2.0, label="Median"
        )

        # percentile bands
        if percentiles:
            ax.plot(
                stats["tmag_center"], stats["p_low"],
                color="orange", lw=1.5,
                label=f"{percentiles[0]}th percentile"
            )
            ax.plot(
                stats["tmag_center"], stats["p_high"],
                color="orange", lw=1.5,
                label=f"{percentiles[1]}th percentile"
            )

            # shaded band
            ax.fill_between(
                stats["tmag_center"],
                stats["p_low"],
                stats["p_high"],
                color="orange", alpha=0.15, linewidth=0
            )

    # Axis labels
    ax.set_xlabel("TESS magnitude (Tmag)")
    ax.set_ylabel(f"{label_text} (ppm)")
    if logy:
        ax.set_yscale("log")

    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{label_text} vs. TESS magnitude")

    ax.legend(frameon=False)
    fig.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=200)
        print(f"Saved plot → {outpath}")
    else:
        plt.show()


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot CDPP vs Tmag with optional median + percentile curves"
    )
    p.add_argument("csv")

    p.add_argument("--cdpp-col", default="rrmscdpp01p0")
    p.add_argument("--tmag-col", default=None,
                   help="Manual override for Tmag column name")

    p.add_argument("--tmag-range", nargs=2, type=float)
    p.add_argument("--clip-percentile", type=float)

    p.add_argument("--logy", action="store_true")
    p.add_argument("--hexbin", action="store_true")
    p.add_argument("--gridsize", type=int, default=50)

    p.add_argument("--bin-width", type=float,
                   help="Tmag bin width for median/percentiles")

    p.add_argument("--percentiles", nargs=2, type=float,
                   metavar=("LOW", "HIGH"),
                   help="Lower and upper percentiles to plot")

    p.add_argument("--title")
    p.add_argument("--out")
    return p.parse_args()


def main():
    args = parse_args()

    # Load
    df = pd.read_csv(args.csv, comment="#")
    df.columns = [normalize_column(c) for c in df.columns]

    # Validate CDPP column
    cdpp_col = normalize_column(args.cdpp_col)
    if cdpp_col not in df.columns:
        raise ValueError(f"CDPP column '{cdpp_col}' not found. Available: {list(df.columns)}")

    # Determine Tmag column
    if args.tmag_col:
        tmag_col = normalize_column(args.tmag_col)
        if tmag_col not in df.columns:
            raise ValueError(f"Tmag column '{tmag_col}' not found. Available: {list(df.columns)}")
    else:
        tmag_col = autodetect_tmag_column(df.columns)
        if tmag_col is None:
            raise ValueError(
                f"Could not auto-detect Tmag column. Available: {list(df.columns)}.\n"
                "Specify manually with --tmag-col."
            )

    print(f"Using Tmag column: {tmag_col}")

    # Filter finite rows
    df = df[np.isfinite(df[tmag_col]) & np.isfinite(df[cdpp_col])].copy()

    # Tmag range
    if args.tmag_range:
        lo, hi = args.tmag_range
        df = df[(df[tmag_col] >= lo) & (df[tmag_col] <= hi)]

    # Clip CDPP
    if args.clip_percentile:
        up = np.percentile(df[cdpp_col], args.clip_percentile)
        df = df[df[cdpp_col] <= up]

    # Percentiles
    percentiles = None
    if args.percentiles:
        p_low, p_high = args.percentiles
        percentiles = (p_low, p_high)

    # Plot
    make_plot(
        df=df,
        tmag_col=tmag_col,
        cdpp_col=cdpp_col,
        logy=args.logy,
        hexbin=args.hexbin,
        gridsize=args.gridsize,
        bin_width=args.bin_width,
        percentiles=percentiles,
        title=args.title,
        outpath=args.out,
    )


if __name__ == "__main__":
    main()
