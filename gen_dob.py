#!/usr/bin/env python3
"""
Learn DOB priors (P(YOB), P(MMDD)) from a dataset and yield YYYYMMDD candidates
in descending probability. Useful for entropy-aware dictionary attacks.

Usage examples:

  # Learn from a CSV with full DOBs (YYYY-MM-DD) and print top 50 DOBs
  python dob_prior_candidates.py --dist-csv dist.csv --dob-col dob --top 50

  # Learn from CSV with only year_of_birth and print top 100 DOBs for 1960..2005
  python dob_prior_candidates.py --dist-csv dist.csv --yob-col year_of_birth \
      --y-start 1960 --y-end 2005 --top 100

  # Save top 10k DOBs to a file (one per line)
  python dob_prior_candidates.py --dist-csv dist.csv --dob-col dob --top 10000 --out top_dobs.txt
"""

import argparse
from datetime import date, datetime
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd


# ------------------------------
# Parsing / Normalization
# ------------------------------
def _parse_dob_to_dt(s: str) -> Optional[pd.Timestamp]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    fmts = ("%Y-%m-%d", "%d.%m.%Y", "%m/%d/%Y", "%Y/%m/%d", "%Y%m%d")
    for f in fmts:
        try:
            return pd.to_datetime(s, format=f, errors="raise")
        except Exception:
            pass
    # last resort: pandas inference (guarded)
    try:
        dt = pd.to_datetime(s, errors="coerce")
        return None if pd.isna(dt) else dt
    except Exception:
        return None


def _is_valid_mmdd(y: int, mmdd: str) -> bool:
    try:
        m = int(mmdd[:2]); d = int(mmdd[2:])
        _ = date(y, m, d)
        return True
    except Exception:
        return False


# ------------------------------
# Learning Priors
# ------------------------------
def learn_dob_priors(
    df: pd.DataFrame,
    dob_col: Optional[str] = None,
    yob_col: Optional[str] = None,
    laplace_alpha: float = 0.0,
) -> Dict[str, Dict[str, float]]:
    """
    Returns a dict with:
      - P_YOB : { '1987': p, '1988': p, ... }
      - P_MMDD: { '0101': p, '1231': p, ... }

    If dob_col not present, P_MMDD will be empty. If neither dob nor yob present,
    returns empty priors.
    """
    priors: Dict[str, Dict[str, float]] = {"P_YOB": {}, "P_MMDD": {}}

    # Build DOB series if we have it
    if dob_col and dob_col in df.columns:
        dob_dt = df[dob_col].map(_parse_dob_to_dt)
        dob_dt = dob_dt[~dob_dt.isna()]
    else:
        dob_dt = pd.Series(dtype="datetime64[ns]")

    # P(YOB)
    yob_series = None
    if len(dob_dt) > 0:
        yob_series = dob_dt.dt.year.astype(int)
    elif yob_col and (yob_col in df.columns):
        # accept ints or strings for year
        try:
            yob_series = df[yob_col].dropna().astype(int)
        except Exception:
            # try to coerce strings like "1987.0"
            yob_series = pd.to_numeric(df[yob_col], errors="coerce").dropna().astype(int)

    if yob_series is not None and len(yob_series) > 0:
        counts = yob_series.value_counts().sort_index()
        if laplace_alpha > 0:
            # Laplace smoothing across observed support
            counts = counts + laplace_alpha
        p_yob = (counts / counts.sum()).to_dict()
        priors["P_YOB"] = {str(int(k)): float(v) for k, v in p_yob.items()}

    # P(MMDD)
    if len(dob_dt) > 0:
        mmdd = dob_dt.dt.strftime("%m%d")
        counts = mmdd.value_counts()
        if laplace_alpha > 0:
            counts = counts + laplace_alpha
        p_mmdd = (counts / counts.sum()).to_dict()
        priors["P_MMDD"] = {str(k): float(v) for k, v in p_mmdd.items()}

    return priors


# ------------------------------
# Candidate Generation
# ------------------------------
_DEFAULT_FALLBACK_MMDD: List[str] = (
    # Spiky/common administrative or memorable dates
    ["0101", "1231", "1111", "0229", "0701", "0704", "0601", "1001"]
    # plus a coarse sweep for coverage (odd months / typical days)
    + [f"{m:02d}{d:02d}" for m in (1, 3, 5, 7, 9, 11) for d in (1, 10, 20, 28)]
)


def dob_candidates(
    priors: Dict[str, Dict[str, float]],
    y_start: Optional[int] = None,
    y_end: Optional[int] = None,
    fallback_mmdd: Optional[List[str]] = None,
) -> Iterator[Tuple[str, float]]:
    """
    Yields (YYYYMMDD, score) sorted by product probability P(YOB) * P(MMDD).
    If P(MMDD) is empty, uses fallback list with uniform mass.

    To keep memory small, we stream YOBs in descending prob and, for each, stream MMDDs
    in descending prob. If you need a materialized top-N list, stop after N yields.
    """
    P_YOB = priors.get("P_YOB", {})
    P_MMDD = priors.get("P_MMDD", {})

    # YOB list
    if P_YOB:
        yob_items = sorted(((int(y), p) for y, p in P_YOB.items()),
                           key=lambda kv: kv[1], reverse=True)
        if y_start is not None or y_end is not None:
            yob_items = [(y, p) for (y, p) in yob_items
                         if (y_start is None or y >= y_start) and (y_end is None or y <= y_end)]
    else:
        # If no YOB prior at all, make a broad uniform range
        ys = range(y_start or 1940, (y_end or 2010) + 1)
        prob = 1.0 / len(list(ys))
        yob_items = [(y, prob) for y in ys]

    # MMDD list
    if P_MMDD:
        mmdd_items = sorted(P_MMDD.items(), key=lambda kv: kv[1], reverse=True)
    else:
        fb = fallback_mmdd or _DEFAULT_FALLBACK_MMDD
        prob = 1.0 / len(fb)
        mmdd_items = [(mmdd, prob) for mmdd in fb]

    for y, py in yob_items:
        for mmdd, pm in mmdd_items:
            if not _is_valid_mmdd(y, mmdd):
                continue
            yield f"{y:04d}{mmdd}", py * pm


# ------------------------------
# CLI
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Learn DOB priors and stream YYYYMMDD candidates by probability.")
    ap.add_argument("--dist-csv", required=True, help="CSV with attribute distribution (DOB and/or YOB).")
    ap.add_argument("--dob-col", default=None, help="Column name with full DOBs (e.g., 'dob').")
    ap.add_argument("--yob-col", default=None, help="Column name with year of birth (e.g., 'year_of_birth').")
    ap.add_argument("--y-start", type=int, default=None, help="Clamp lower YOB (e.g., 1960).")
    ap.add_argument("--y-end", type=int, default=None, help="Clamp upper YOB (e.g., 2005).")
    ap.add_argument("--laplace", type=float, default=0.0, help="Laplace smoothing alpha (e.g., 0.5).")
    ap.add_argument("--top", type=int, default=50, help="How many DOB candidates to print/save.")
    ap.add_argument("--out", default="", help="Optional path to save top candidates (one YYYYMMDD per line).")
    args = ap.parse_args()

    df = pd.read_csv(args.dist_csv)
    priors = learn_dob_priors(df, dob_col=args.dob_col, yob_col=args.yob_col, laplace_alpha=args.laplace)

    if not priors["P_YOB"] and not priors["P_MMDD"]:
        print("WARNING: No DOB or YOB information found; using uniform fallbacks.")

    # Stream top-N
    out_rows: List[str] = []
    for i, (dob, score) in enumerate(dob_candidates(priors, args.y_start, args.y_end)):
        if i >= args.top:
            break
        print(f"{i+1:4d}. {dob}   score={score:.8f}")
        out_rows.append(dob)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for dob in out_rows:
                f.write(dob + "\n")
        print(f"\nSaved top {len(out_rows)} DOB candidates to: {args.out}")


if __name__ == "__main__":
    main()
