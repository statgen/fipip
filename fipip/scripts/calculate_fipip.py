#!/usr/bin/env python3
"""
LOCO (Leave-One-Chromosome-Out) XGBoost with fiPIP generation, relaxed for chromosomes
that are not present in training labels.

Behavior:
- For chromosome c in the prediction file:
  * If c exists among labeled rows in training: train on all labeled rows where chrom != c (true LOCO).
  * If c does NOT exist among labeled rows: train on ALL labeled rows (no exclusion) and use that model.

This version is NAME-AWARE for non-predictor columns:
- Excludes from features: {'variant', 'label', 'cs_id', 'pip', <chrom-col>} if present.
- The remaining columns, in order, are treated as quantitative score columns.
- Score column names and order must match between training and prediction files.

Outputs:
  1) <outdir>/<model_name>.predictions.tsv
  2) <outdir>/<model_name>.models/chr<chrom>.xgb.json (one per chromosome in prediction file)
     plus <outdir>/<model_name>.models/manifest.json
"""

import argparse
import os
import sys
import re
import json
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import gzip

try:
    import xgboost as xgb
except ImportError:
    sys.stderr.write("ERROR: xgboost is not installed. Try: pip install xgboost\n")
    raise


# ---------------------------
# I/O helpers
# ---------------------------

def infer_sep(path: str) -> str:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if "\t" in s:
                return "\t"
            if "," in s:
                return ","
            return r"\s+"
    return ","

def check_binary_series(s: pd.Series, name: str = "label") -> None:
    vals = s.dropna().unique()
    bad = [v for v in vals if v not in (0, 1, 0.0, 1.0)]
    if bad:
        raise ValueError(f"{name} must be binary (0/1). Found: {bad[:5]}...")


def read_table(path: str) -> pd.DataFrame:
    sep = infer_sep(path)
    df = pd.read_csv(path, sep=sep, engine="python")
    return df


def derive_score_columns(
    df: pd.DataFrame,
    chrom_col: str,
    is_training: bool,
) -> Tuple[List[str], List[str]]:
    """
    Return (score_columns, excluded_columns_seen).

    Exclude by NAME any of:
      - 'variant'
      - 'label' (training only)
      - 'cs_id', 'pip' (prediction only)
      - chrom_col (always excluded from predictors)

    The rest (in original order) are treated as quantitative score columns.
    """
    required_exclude = {"variant", chrom_col}
    if is_training:
        required_exclude |= {"label"}
    else:
        required_exclude |= {"cs_id", "pip"}

    present_excludes = [c for c in df.columns if c in required_exclude]
    score_cols = [c for c in df.columns if c not in required_exclude]
    return score_cols, present_excludes


def select_features_matrix(
    df: pd.DataFrame,
    score_cols: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Build feature matrix from score columns.
    Apply absolute value transform (to mirror R code behavior).
    """
    numeric_df = df[score_cols].apply(pd.to_numeric, errors="raise")
    X = np.abs(numeric_df.values)
    return X, score_cols


# ---------------------------
# Modeling helpers
# ---------------------------

def train_xgb_classifier(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int = 3,
    eta: float = 0.1,
    nrounds: int = 100,
    nthread: int = 0,
    seed: int = 123,
) -> xgb.Booster:
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": max_depth,
        "eta": eta,
        "nthread": nthread,
        "seed": seed,
    }
    booster = xgb.train(params, dtrain, num_boost_round=nrounds, verbose_eval=False)
    return booster


def predict_proba(booster: xgb.Booster, X: np.ndarray) -> np.ndarray:
    dtest = xgb.DMatrix(X)
    return booster.predict(dtest)


def compute_fipip(
    df_pred: pd.DataFrame,
    pred_col: str,
    cs_col: str,
    pip_col: str,
) -> pd.Series:
    """
    fiPIP = normalize( prediction * normalize(PIP within credible set) ) within each cs_id.
    If sum within a cs is zero, fiPIP = 0 for that cs.
    """
    pip_sum = df_pred.groupby(cs_col, observed=True)[pip_col].transform("sum")
    pip_norm = np.where(pip_sum > 0, df_pred[pip_col] / pip_sum, 0.0)
    raw = df_pred[pred_col].values * pip_norm
    denom = (
        df_pred.assign(_raw=raw)
        .groupby(cs_col, observed=True)["_raw"]
        .transform("sum")
        .values
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        fi = np.where(denom > 0, raw / denom, 0.0)
    return pd.Series(fi, index=df_pred.index, name="fiPIP")


# ---------------------------
# Main (LOCO with relaxed rule)
# ---------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "LOCO XGBoost: Train one model per chromosome (held out), "
            "predict on file #2 per chromosome, and compute fiPIPs. "
            "If a chromosome in the prediction file is absent from training labels, "
            "train on ALL labeled rows (no exclusion) for that chromosome."
        )
    )
    ap.add_argument("--train-file", required=True, help="Training file (must include 'variant', 'label', and --chrom-col).")
    ap.add_argument("--test-file", required=True, help="Prediction file (must include 'variant', 'cs_id', 'pip', and --chrom-col).")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--model-name", default="xgb_loco", help="Base name for outputs (no extension).")
    ap.add_argument("--chrom-col", default="chr", help="Column name for chromosome in BOTH files (e.g., 'chr').")
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--nrounds", type=int, default=100)
    ap.add_argument("--nthread", type=int, default=0)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    models_dir = os.path.join(args.outdir, f"{args.model_name}.models")
    os.makedirs(models_dir, exist_ok=True)

    # Load data
    train_df = read_table(args.train_file)
    pred_df = read_table(args.test_file)

    # Required columns present?
    for col in ["variant", "label"]:
        if col not in train_df.columns:
            raise ValueError(f"Training file missing required column '{col}'.")
    for col in ["variant", "cs_id", "pip"]:
        if col not in pred_df.columns:
            raise ValueError(f"Prediction file missing required column '{col}'.")
    if args.chrom_col not in train_df.columns:
        raise ValueError(f"Training file missing chromosome column '{args.chrom_col}'.")
    if args.chrom_col not in pred_df.columns:
        raise ValueError(f"Prediction file missing chromosome column '{args.chrom_col}'.")

    # Validate binary labels
    check_binary_series(train_df["label"], "label")
    # Ensure pip numeric
    if not np.issubdtype(pred_df["pip"].dtype, np.number):
        pred_df["pip"] = pd.to_numeric(pred_df["pip"], errors="coerce")
    if pred_df["pip"].isna().any():
        raise ValueError("Column 'pip' must be numeric (no NA after coercion).")

    # Derive score columns by NAME-exclusion
    train_scores, _train_ex = derive_score_columns(train_df, chrom_col=args.chrom_col, is_training=True)
    pred_scores,  _pred_ex  = derive_score_columns(pred_df,  chrom_col=args.chrom_col, is_training=False)

    # Score names must match exactly (names + order)
    if train_scores != pred_scores:
        train_only = [c for c in train_scores if c not in pred_scores]
        pred_only  = [c for c in pred_scores  if c not in train_scores]
        raise ValueError(
            "Quantitative score column names do NOT match between files.\n"
            f"- Train-only (first few): {train_only[:10]}\n"
            f"- Predict-only (first few): {pred_only[:10]}\n"
            "Fix names/order so they are identical."
        )

    # Training rows with non-NA labels
    train_mask = train_df["label"].notna()
    if train_mask.sum() == 0:
        raise ValueError("No non-NA labels found in training file.")
    train_df_lab = train_df.loc[train_mask].copy()
    train_df_lab["label"] = train_df_lab["label"].astype(int)

    # Chromosomes present
    chroms_pred = list(pd.unique(pred_df[args.chrom_col]))
    chroms_train = set(pd.unique(train_df_lab[args.chrom_col]))

    # Collect predictions per chromosome
    preds_frames = []
    manifest: Dict[str, dict] = {
        "model_name": args.model_name,
        "train_file": os.path.abspath(args.train_file),
        "test_file": os.path.abspath(args.test_file),
        "chrom_col": args.chrom_col,
        "used_score_columns": None,  # filled after first build
        "total_score_columns": train_scores,
        "xgboost_params": {
            "max_depth": args.max_depth,
            "eta": args.eta,
            "nrounds": args.nrounds,
            "nthread": args.nthread,
            "seed": args.seed,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        },
        "models": [],  # list of {"chrom": "...", "path": "...", "n_train": int, "mode": "loco"|"all"}
    }

    for chrom in chroms_pred:
        # Choose training subset according to relaxed LOCO rule
        if chrom in chroms_train:
            # True LOCO: exclude this chromosome from training
            train_loco = train_df_lab[train_df_lab[args.chrom_col] != chrom]
            mode = "loco"
        else:
            # Chrom not in labeled training => train on ALL labeled rows
            train_loco = train_df_lab
            mode = "all"

        if train_loco.empty:
            raise ValueError(f"Training set is empty for chrom={chrom} (mode={mode}).")

        y = train_loco["label"].to_numpy()
        X_train, used_cols = select_features_matrix(train_loco, train_scores)
        if manifest["used_score_columns"] is None:
            manifest["used_score_columns"] = used_cols

        booster = train_xgb_classifier(
            X_train, y,
            max_depth=args.max_depth,
            eta=args.eta,
            nrounds=args.nrounds,
            nthread=args.nthread,
            seed=args.seed,
        )

        # Save per-chrom model
        model_path = os.path.join(models_dir, f"{chrom}.xgb.json")
        booster.save_model(model_path)
        manifest["models"].append({
            "chrom": str(chrom),
            "path": model_path,
            "n_train": int(len(y)),
            "mode": mode
        })

        # Predict on prediction rows where chrom==held-out chrom
        pred_block = pred_df[pred_df[args.chrom_col] == chrom].copy()
        if pred_block.empty:
            continue
        X_pred, _ = select_features_matrix(pred_block, pred_scores)
        pred_block["prediction"] = predict_proba(booster, X_pred)

        preds_frames.append(pred_block)

    if not preds_frames:
        raise ValueError("No predictions were generated (check chromosome values).")

    all_pred = pd.concat(preds_frames, axis=0, ignore_index=False).copy()

    # Compute fiPIPs normalized within each credible set
    all_pred["fiPIP"] = compute_fipip(all_pred, pred_col="prediction", cs_col="cs_id", pip_col="pip")

    # Outputs
    pred_out = os.path.join(args.outdir, f"{args.model_name}.predictions.tsv")
    manifest_out = os.path.join(models_dir, "manifest.json")

    # Keep original columns first, then add prediction & fiPIP
    base_cols = [c for c in all_pred.columns if c not in ("prediction", "fiPIP")]
    ordered_cols = base_cols + ["prediction", "fiPIP"]
    all_pred[ordered_cols].to_csv(pred_out, sep="\t", index=False)

    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] Wrote predictions: {pred_out}")
    print(f"[OK] Saved per-chrom models in: {models_dir}")
    print(f"[OK] Wrote model manifest: {manifest_out}")
    # Helpful note if any "all" mode was used
    any_all = any(m["mode"] == "all" for m in manifest["models"])
    if any_all:
        print("[NOTE] Some chromosomes were not present in training labels; for those, the model was trained on ALL labeled rows (not strictly LOCO).")


if __name__ == "__main__":
    main()

def calculate_fipip(argv=None):
    return main()
