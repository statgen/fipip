#!/usr/bin/env python3
# borzoi_2.py
#
# Scans an input directory for *_wt.obj files, pairs each with *_mut.obj,
# reproduces the exon-masked delta computation, and writes a CSV.
#
# Inputs: directory of .wt.obj / .mut.obj pickles produced by borzoi_1.py
# Outputs: CSV with one row per variant (and gene if provided), columns for selected tracks.
#
# Parity details with R (see README in the source message):
# - Assumes pickle shapes are (1, 4, 16352, 89).

import argparse
import csv
import glob
import os
import pickle
import sys
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd


# -----------------------------
# Hard-coded constants (edit if needed)
# -----------------------------
GTF_PATH = None  # Hard code in \path\to\GTF.gtf here or include with --gtf-path

# Default model/geometry parameters, please feel free to change if desired
SEQ_LEN = 524288
NORMALIZE_WINDOW = 4096
BIN_SIZE = 32
PAD = 16

# Track rescaling parameters, please feel free to change if desired
RESCALE_TRACKS = True
SCALE_PARAMETER = 0.01
CLIP_SOFT = 384
TRACK_TRANSFORM = 0.75


# -----------------------------
# Utilities
# -----------------------------
def parse_track_indices(spec: str, n_tracks: int = 89) -> List[int]:
    """
    Parse a track index spec into 0-based indices.
    Examples:
      "1-89"
      "1,5,7-10,23,80-89"
    Tracks are 1-indexed in the spec; we convert to 0-index.
    """
    if not spec:
        return list(range(n_tracks))
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a_i = int(a)
            b_i = int(b)
            if a_i < 1 or b_i > n_tracks or a_i > b_i:
                raise ValueError(f"Invalid track range: {part}")
            out.update(range(a_i - 1, b_i))  # inclusive upper, converted to 0-based
        else:
            v = int(part)
            if v < 1 or v > n_tracks:
                raise ValueError(f"Invalid track index: {v}")
            out.add(v - 1)
    return sorted(out)


def load_pickle(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        return pickle.load(f)


def _collapse_mean(arr: np.ndarray) -> np.ndarray:
    """
    Collapse Borzoi predictions from shape (1, 4, 16352, 89) to (16352, 89)
    by averaging over the fold dimension.
    If the array already has shape (16352, 89), it is returned as-is.
    Accepts a small variety of close shapes and tries to do the right thing.
    """
    a = np.asarray(arr)
    if a.ndim == 4:
        d1, d2, d3, d4 = a.shape
        if d1 != 1:
            a0 = a.mean(axis=0)  # reduce d1
            return a0.mean(axis=0)  # reduce d2 -> (d3, d4)
        else:
            return a[0].mean(axis=0)
    elif a.ndim == 3 and a.shape[0] == 4:
        return a.mean(axis=0)
    elif a.ndim == 2:
        return a
    else:
        raise ValueError(f"Unexpected prediction array shape: {a.shape}")


def _apply_rescale(a: np.ndarray,
                   rescale_tracks: bool = RESCALE_TRACKS,
                   scale_parameter: float = SCALE_PARAMETER,
                   clip_soft: Optional[float] = CLIP_SOFT,
                   track_transform: float = TRACK_TRANSFORM) -> np.ndarray:
    """
    Mirror R rescaling:
      a <- a / scale_parameter
      if (!is.na(clip_soft)) {
         a_unclipped <- ((a - clip_soft)^2) + clip_soft
         a[a > clip_soft] <- a_unclipped[a > clip_soft]
      }
      a <- a ^ (1 / track_transform)
      a <- log1p(a)
    """
    x = a.astype(np.float64, copy=True)
    if rescale_tracks:
        x /= float(scale_parameter)

        if clip_soft is not None:
            mask = x > clip_soft
            x_masked = x[mask] - clip_soft
            x_masked = x_masked * x_masked + clip_soft
            x[mask] = x_masked

        x = np.power(x, 1.0 / float(track_transform))

    x = np.log1p(x)
    return x


def _build_exon_mask(gtf_df: Optional[pd.DataFrame],
                     chrom: str,
                     center_pos: int,
                     seq_len: int,
                     gene_name: Optional[str]) -> np.ndarray:
    """
    Return a (16384,) binary mask (0/1) at BIN_SIZE=32 resolution over the full window,
    then trim 16 bins on each side to length 16352
    If no exons overlap (or gene not provided / not matched), fall back to the R default:
      c(rep(0,7209), rep(1,1966), rep(0,7209))
    """
    if gtf_df is None:
        return _default_mask_trimmed()

    start = center_pos - seq_len // 2
    end = center_pos + seq_len // 2  # exclusive end

    gtf = gtf_df
    gtf = gtf[(gtf["seq"] == "exon") &
              (gtf["chr"] == chrom) &
              (gtf["end1"] >= max(0, start)) &
              (gtf["start1"] <= end)]

    if gene_name is not None and len(gene_name) > 0:
        gtf = gtf[gtf["gene"] == gene_name]

    if gtf.shape[0] == 0:
        return _default_mask_trimmed()

    bin_idxs = []
    for _, row in gtf.iterrows():
        s = int(row["start1"])
        e = int(row["end1"])  # inclusive in GTF
        s_clip = max(s, start)
        e_clip = min(e, end - 1)
        if s_clip > e_clip:
            continue
        rel_s = s_clip - start
        rel_e = e_clip - start
        b0 = rel_s // BIN_SIZE
        b1 = rel_e // BIN_SIZE
        if b0 < 0:
            b0 = 0
        if b1 > 16383:
            b1 = 16383
        if b0 <= b1:
            bin_idxs.append((b0, b1))

    base = np.zeros(16384, dtype=np.float64)
    for b0, b1 in bin_idxs:
        base[b0:b1 + 1] = 1.0

    return base[16:16368]


def _read_gtf(gtf_path: str) -> pd.DataFrame:
    """
    Read and lightly normalize the GTF into:
      chr, annotation, seq, start1, end1, dot1, strand, dot2, gene
    """
    colnames = ["chr", "annotation", "seq", "start1", "end1", "dot1", "strand", "dot2", "gene"]
    df = pd.read_csv(gtf_path, sep="\t", header=None, comment="#", names=colnames, dtype=str)

    df["start1"] = df["start1"].astype(int)
    df["end1"] = df["end1"].astype(int)

    def parse_gene(attr: str) -> str:
        s = str(attr)
        if 'gene_name "' in s:
            try:
                left = s.split('gene_name "', 1)[1]
                token = left.split('"', 1)[0]
                return token.split(".")[0]
            except Exception:
                pass
        if '"' in s:
            try:
                token = s.split('"', 2)[1]
                return token.split(".")[0]
            except Exception:
                pass
        return s

    df["gene"] = df["gene"].map(parse_gene)
    return df


def _extract_tokens_from_basename(basename: str) -> Tuple[str, int, Optional[str], Optional[str]]:
    """
    Expect filenames like: chr1_123456_A_T_wt.obj (from borzoi_1.py)
    Returns (chrom, center_pos:int, ref:str|None, alt:str|None).
    """
    stem = os.path.splitext(basename)[0]
    if stem.endswith("_wt"):
        stem = stem[:-3]
    elif stem.endswith("_mut"):
        stem = stem[:-4]
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected basename format: {basename}")
    chrom = parts[0]
    try:
        center_pos = int(parts[1])
    except Exception:
        raise ValueError(f"Cannot parse center position from {basename}")
    ref = parts[2] if len(parts) >= 4 else None
    alt = parts[3] if len(parts) >= 4 else None
    return chrom, center_pos, ref, alt


def compute_delta_for_pair(wt_path: str,
                           mut_path: str,
                           track_indices_0based: Sequence[int],
                           gtf_df: Optional[pd.DataFrame],
                           default_gene: Optional[str],
                           rescale_tracks: bool,
                           scale_parameter: float,
                           clip_soft: Optional[float],
                           track_transform: float) -> Tuple[str, np.ndarray]:
    """
    Core routine:
      - load pickles
      - rescale/transform/log1p
      - collapse → (16352, 89)
      - build exon mask from GTF (+ fallback)
      - mask rows, column-sum, compute (mut - wt)
      - subset tracks
      Returns (row_id, vector)
    """
    base = os.path.basename(wt_path)
    chrom, center_pos, _, _ = _extract_tokens_from_basename(base)

    wt_raw = load_pickle(wt_path)
    mut_raw = load_pickle(mut_path)

    wt_scaled = _apply_rescale(wt_raw, rescale_tracks, scale_parameter, clip_soft, track_transform)
    mut_scaled = _apply_rescale(mut_raw, rescale_tracks, scale_parameter, clip_soft, track_transform)

    wt_mat = _collapse_mean(wt_scaled)
    mut_mat = _collapse_mean(mut_scaled)

    if wt_mat.shape[0] != 16352 or wt_mat.shape[1] != 89:
        raise ValueError(f"{base}: unexpected collapsed shape {wt_mat.shape}, expected (16352, 89)")

    mask = _build_exon_mask(gtf_df, chrom, center_pos, SEQ_LEN, default_gene)

    wt_masked = wt_mat * mask[:, None]
    mut_masked = mut_mat * mask[:, None]

    wt_sums = wt_masked.sum(axis=0)
    mut_sums = mut_masked.sum(axis=0)

    delta = mut_sums - wt_sums  # (89,)
    delta_sub = delta[np.array(track_indices_0based, dtype=int)]

    row_id = os.path.splitext(os.path.basename(base))[0].replace("_wt", "")
    return row_id, delta_sub


def find_pairs(indir: str) -> List[Tuple[str, str]]:
    """
    Find *_wt.obj files in indir that have matching *_mut.obj neighbors.
    Return list of (wt_path, mut_path).
    """
    wt_files = sorted(glob.glob(os.path.join(indir, "*_wt.obj")))
    pairs = []
    for wt in wt_files:
        mut = wt.replace("_wt.obj", "_mut.obj")
        if os.path.exists(mut):
            pairs.append((wt, mut))
    return pairs


def _default_mask_trimmed() -> np.ndarray:
    base = np.zeros(16384, dtype=np.float64)
    base[7209:7209 + 1966] = 1.0
    return base[16:16368]


def _read_gene_map_autodetect(path: str) -> pd.DataFrame:
    """
    Read a two-column mapping file into columns ['row_id','gene'] while
    preserving row order and allowing either comma-separated or whitespace-separated formats.
    - If the first non-empty, non-comment line contains a comma, use sep=",".
    - Otherwise, use sep=r"\\s+" (tabs/spaces).
    Only the first two columns are read; extra columns are ignored.
    """
    sep = r"\s+"
    # Peek first meaningful line to decide
    try:
        with open(path, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if "," in s:
                    sep = ","
                break
    except Exception as e:
        raise RuntimeError(f"Failed to inspect gene-map file '{path}': {e}")

    try:
        df = pd.read_csv(path, sep=sep, header=None, usecols=[0, 1], names=["row_id", "gene"])
    except Exception as e:
        raise RuntimeError(f"Failed to read gene-map '{path}' with sep='{sep}': {e}")

    # Normalize whitespace and types
    df["row_id"] = df["row_id"].astype(str).str.strip()
    df["gene"] = df["gene"].astype(str).str.strip()
    # Drop completely empty row_id entries
    df = df[df["row_id"].str.len() > 0].reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate Borzoi predictions for WT/Mut .obj pairs into exon-masked deltas per track."
    )
    parser.add_argument(
        "-i", "--input-dir", required=True,
        help="Directory containing *_wt.obj and *_mut.obj files (e.g., output of borzoi_1.py)."
    )
    parser.add_argument(
        "-t", "--tracks", default="1-89",
        help="Track indices (1-89) to keep. Examples: '1-89' (default), '1,5,7-10,80-89'."
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Path to write CSV. Rows = variants (and genes), columns = selected tracks."
    )
    parser.add_argument(
        "--gtf-path", default=GTF_PATH,
        help="Path to gencode41_basic_nort.gtf. If omitted or --no-gtf is set, uses default center window mask."
    )
    parser.add_argument(
        "--no-gtf", action="store_true",
        help="Ignore any GTF and always use the default center window mask."
    )
    parser.add_argument(
        "--gene", default=None,
        help="Optional constant gene symbol to use for all variants (e.g., 'BRCA1'). "
             "If omitted and no gene-map is provided, mask falls back to the default center window."
    )
    parser.add_argument(
        "--gene-map", default=None,
        help="Optional two-column mapping file: row_id,gene. "
             "Auto-detects comma-separated or whitespace-separated formats. "
             "May contain multiple rows for the same row_id (duplicates allowed). "
             "Output preserves map order for these rows."
    )
    parser.add_argument(
        "--no-rescale", default=RESCALE_TRACKS, action="store_true",
        help="Disable the track rescaling/soft-clip/inverse-transform step (still applies log1p)."
    )
    parser.add_argument(
        "--scale-parameter", type=float, default=SCALE_PARAMETER,
        help=f"Scale parameter used when rescaling (default {SCALE_PARAMETER})."
    )
    parser.add_argument(
        "--clip-soft", type=float, default=CLIP_SOFT,
        help=f"Soft clip threshold (default {CLIP_SOFT}); set to a negative value to disable."
    )
    parser.add_argument(
        "--track-transform", type=float, default=TRACK_TRANSFORM,
        help=f"Inverse power transform parameter (default {TRACK_TRANSFORM})."
    )

    args = parser.parse_args()

    # Resolve rescaling options
    rescale_tracks = not args.no_rescale
    clip_soft = None if (args.clip_soft is None or args.clip_soft < 0) else float(args.clip_soft)

    # Parse track spec
    track_ix = parse_track_indices(args.tracks, n_tracks=89)

    # Load GTF if used
    gtf_df = None
    if not args.no_gtf:
        if args.gtf_path and os.path.exists(args.gtf_path):
            gtf_df = _read_gtf(args.gtf_path)
        else:
            print(f"[INFO] No GTF found at {args.gtf_path}; falling back to default center window mask.", file=sys.stderr)

    # Discover available file pairs
    pairs = find_pairs(args.input_dir)
    if not pairs:
        sys.exit(f"No *_wt.obj with matching *_mut.obj found in {args.input_dir}")

    # Map row_id -> (wt_path, mut_path)
    id_to_pair: Dict[str, Tuple[str, str]] = {}
    for wt_path, mut_path in pairs:
        rid = os.path.splitext(os.path.basename(wt_path))[0].replace("_wt", "")
        id_to_pair[rid] = (wt_path, mut_path)

    rows = []
    colnames = [f"track_{i+1}" for i in track_ix]

    # If gene-map provided, process exactly in the mapping order (duplicates allowed)
    processed_ids_from_map = set()
    if args.gene_map:
        gm_df = _read_gene_map_autodetect(args.gene_map)
        for _, r in gm_df.iterrows():
            rid = r["row_id"]
            gene_sym = None if pd.isna(r["gene"]) or r["gene"] == "" else str(r["gene"])
            if rid not in id_to_pair:
                print(f"[WARN] gene-map row_id '{rid}' not found among input files; skipping.", file=sys.stderr)
                continue
            wt_path, mut_path = id_to_pair[rid]
            try:
                # Compute directly with the provided gene (can be None/empty)
                _, vec = compute_delta_for_pair(
                    wt_path=wt_path,
                    mut_path=mut_path,
                    track_indices_0based=track_ix,
                    gtf_df=gtf_df,
                    default_gene=gene_sym,
                    rescale_tracks=rescale_tracks,
                    scale_parameter=float(args.scale_parameter),
                    clip_soft=clip_soft,
                    track_transform=float(args.track_transform),
                )
                rows.append((rid, gene_sym if gene_sym is not None else "", *vec.tolist()))
            except Exception as e:
                print(f"[WARN] Skipping {wt_path} / {mut_path} for gene '{gene_sym}': {e}", file=sys.stderr)
            processed_ids_from_map.add(rid)

    # For any remaining pairs not mentioned in the gene-map, process once each
    for rid, (wt_path, mut_path) in id_to_pair.items():
        if args.gene_map and rid in processed_ids_from_map:
            continue
        try:
            gene_for_this = args.gene  # constant gene if provided
            _, vec = compute_delta_for_pair(
                wt_path=wt_path,
                mut_path=mut_path,
                track_indices_0based=track_ix,
                gtf_df=gtf_df,
                default_gene=gene_for_this,
                rescale_tracks=rescale_tracks,
                scale_parameter=float(args.scale_parameter),
                clip_soft=clip_soft,
                track_transform=float(args.track_transform),
            )
            rows.append((rid, gene_for_this if gene_for_this is not None else "", *vec.tolist()))
        except Exception as e:
            print(f"[WARN] Skipping {wt_path} / {mut_path}: {e}", file=sys.stderr)

    if not rows:
        sys.exit("No rows produced; nothing to write.")

    # Write CSV (always include 'gene' column to disambiguate duplicates)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "gene"] + colnames)
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
