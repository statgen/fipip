#!/usr/bin/env python3
"""
Prediction-only script for fiPIP generation using saved per-chromosome XGBoost models.

- Reads models from --models-dir (expects .xgb.json files, optionally manifest.json).
- Reads a prediction file with columns:
    variant, cs_id, pip, <chrom-col>, <score1...>
- Builds features from either:
    * manifest.json's "used_score_columns" (preferred), or
    * by excluding non-predictor columns by NAME: {'variant','cs_id','pip',<chrom-col>}
- Predicts per chromosome using the matching model file:
    * If predicted chromosome 'X' has a model file named 'X.xgb.json' -> use it
    * Otherwise:
         - default: error
         - with --fallback-chrom <Y>: use model 'Y.xgb.json' for all missing chromosomes
- Computes fiPIPs by PIP-normalize then multiply by predictions, then renormalize within cs_id.
- Writes:
    <outdir>/<model_name>.predictions.tsv
"""

import argparse
import os
import sys
import json
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    sys.stderr.write("ERROR: xgboost is not installed. Try: pip install xgboost\n")
    raise


# ---------------------------
# Helpers
# ---------------------------

def infer_sep(path: str) -> str:
    """Infer CSV/TSV delimiter using first non-empty line."""
    with open(path, "r", encoding="utf-8") as f:
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


def read_table(path: str) -> pd.DataFrame:
    sep = infer_sep(path)
    return pd.read_csv(path, sep=sep, engine="python")


def compute_fipip(df_pred: pd.DataFrame, pred_col: str, cs_col: str, pip_col: str) -> pd.Series:
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


def load_manifest(models_dir: str) -> Optional[dict]:
    manifest_path = os.path.join(models_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def list_model_files(models_dir: str) -> Dict[str, str]:
    """
    Return mapping {chrom_value -> path_to_model_file}, where chrom_value is the
    base filename without extension, e.g. 'chr1' for 'chr1.xgb.json'.
    """
    out = {}
    for fn in os.listdir(models_dir):
        if not fn.endswith(".xgb.json"):
            continue
        base = fn[:-len(".xgb.json")]
        out[base] = os.path.join(models_dir, fn)
    return out


def derive_score_columns_from_file(
    df: pd.DataFrame,
    chrom_col: str,
) -> List[str]:
    """
    Derive score columns by excluding non-predictor columns by name.
    """
    exclude = {"variant", "cs_id", "pip", chrom_col}
    return [c for c in df.columns if c not in exclude]


def build_feature_matrix(
    df: pd.DataFrame,
    score_cols: List[str],
    abs_transform: bool = True,
) -> np.ndarray:
    """
    Build numeric feature matrix from given score columns; apply abs() if requested.
    """
    numeric_df = df[score_cols].apply(pd.to_numeric, errors="raise")
    X = numeric_df.values
    if abs_transform:
        X = np.abs(X)
    return X


def predict_block(booster: xgb.Booster, X: np.ndarray) -> np.ndarray:
    dtest = xgb.DMatrix(X)
    return booster.predict(dtest)


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Apply saved per-chromosome XGBoost models to a prediction file and compute fiPIPs."
        )
    )
    ap.add_argument("--predict-file", required=True, help="Prediction file with columns: variant, cs_id, pip, <chrom-col>, scores...")
    ap.add_argument("--models-dir", required=True, help="Directory containing .xgb.json models (and optional manifest.json).")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--model-name", default="xgb_apply", help="Base name for output file (no extension).")
    ap.add_argument("--chrom-col", default="chrom", help="Column name for chromosome in prediction file (e.g., 'chr').")
    ap.add_argument("--fallback-chrom", default=None,
                    help="If a chromosome has no matching model file, use this chromosome's model instead (e.g., 'chr1'). Default: error.")
    ap.add_argument("--no-abs", action="store_true",
                    help="Disable absolute-value transform on features (defaults to applying abs()).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load prediction table
    df = read_table(args.predict_file)

    # Basic checks
    for col in ["variant", "cs_id", "pip"]:
        if col not in df.columns:
            raise ValueError(f"Prediction file missing required column '{col}'.")
    if args.chrom_col not in df.columns:
        raise ValueError(f"Prediction file missing chromosome column '{args.chrom_col}'.")
    # pip numeric
    if not np.issubdtype(df["pip"].dtype, np.number):
        df["pip"] = pd.to_numeric(df["pip"], errors="coerce")
    if df["pip"].isna().any():
        raise ValueError("Column 'pip' must be numeric (no NA after coercion).")

    # Load manifest if present
    manifest = load_manifest(args.models_dir)
    if manifest and "used_score_columns" in manifest and manifest["used_score_columns"]:
        score_cols = manifest["used_score_columns"]
        # verify they exist in df
        missing = [c for c in score_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Prediction file is missing score columns from manifest: {missing[:10]} ..."
            )
    else:
        # Derive from file (name-aware exclusion)
        score_cols = derive_score_columns_from_file(df, chrom_col=args.chrom_col)

    # Build a model map from files
    model_map = list_model_files(args.models_dir)
    if not model_map:
        raise ValueError(f"No '.xgb.json' model files found in: {args.models_dir}")

    # Check fallback model availability if specified
    if args.fallback_chrom:
        if args.fallback_chrom not in model_map:
            raise ValueError(
                f"--fallback-chrom '{args.fallback_chrom}' has no model file in {args.models_dir}. "
                f"Expected: {args.fallback_chrom}.xgb.json"
            )

    # Predict per-chromosome blocks
    chrom_values = list(pd.unique(df[args.chrom_col]))
    pred_frames = []
    used_models_record = []

    for chrom in chrom_values:
        chrom_str = str(chrom)
        # Choose model
        if chrom_str in model_map:
            model_path = model_map[chrom_str]
            chosen = chrom_str
        elif args.fallback_chrom is not None:
            model_path = model_map[args.fallback_chrom]
            chosen = args.fallback_chrom
        else:
            raise ValueError(
                f"No model file for chromosome '{chrom_str}'. "
                f"Expected {chrom_str}.xgb.json in {args.models_dir} "
                f"(or use --fallback-chrom <chrom>)."
            )

        # Load model
        booster = xgb.Booster()
        booster.load_model(model_path)

        # Slice rows for this chromosome
        block = df[df[args.chrom_col] == chrom_str].copy()
        if block.empty:
            continue

        # Features
        X = build_feature_matrix(block, score_cols, abs_transform=(not args.no_abs))
        # Predict
        block["prediction"] = predict_block(booster, X)

        pred_frames.append(block)
        used_models_record.append({"chrom": chrom_str, "model_used": os.path.basename(model_path)})

    if not pred_frames:
        raise ValueError("No predictions generated (check chromosome values and model filenames).")

    all_pred = pd.concat(pred_frames, axis=0, ignore_index=False).copy()

    # Compute fiPIPs
    all_pred["fiPIP"] = compute_fipip(all_pred, pred_col="prediction", cs_col="cs_id", pip_col="pip")

    # Output
    pred_out = os.path.join(args.outdir, f"{args.model_name}.predictions.tsv")
    base_cols = [c for c in all_pred.columns if c not in ("prediction", "fiPIP")]
    ordered = base_cols + ["prediction", "fiPIP"]
    all_pred[ordered].to_csv(pred_out, sep="\t", index=False)

    # Write a small apply-manifest
    apply_manifest = {
        "predict_file": os.path.abspath(args.predict_file),
        "models_dir": os.path.abspath(args.models_dir),
        "chrom_col": args.chrom_col,
        "score_columns_source": "manifest.used_score_columns" if (manifest and "used_score_columns" in manifest and manifest["used_score_columns"]) else "derived_from_prediction_file",
        "used_score_columns": score_cols,
        "abs_transform": not args.no_abs,
        "fallback_chrom": args.fallback_chrom,
        "models_used": used_models_record,
        "output_predictions": os.path.abspath(pred_out),
    }
    with open(os.path.join(args.outdir, f"{args.model_name}.apply_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(apply_manifest, f, indent=2)

    print(f"[OK] Wrote predictions: {pred_out}")
    print(f"[OK] Wrote apply manifest: {os.path.join(args.outdir, f'{args.model_name}.apply_manifest.json')}")
    print(f"[INFO] Models used per chromosome: {used_models_record}")


if __name__ == "__main__":
    main()
