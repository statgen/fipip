#!/usr/bin/env python3
# borzoi_2.py
#
# Scans an input directory for *_wt.obj files, pairs each with *_mut.obj,
# reproduces the exon-masked delta computation, and writes a CSV.
#
# Inputs: directory of .wt.obj / .mut.obj pickles produced by borzoi_1.py
# Outputs: CSV with one row per variant, columns for selected tracks.
#
# Parity details with R (see README in the source message):
# - Assumes pickle shapes are (1, 4, 16352, 89).

import argparse
import csv
import glob
import os
import pickle
import sys
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Hard-coded constants (edit if needed)
# -----------------------------
GTF_PATH = "" # Hard code in \path\to\GTF.gtf here or include with --gtf-path

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
            # If d1 is not 1 but 4, we can also average over that leading axis
            # to emulate "use index 0" from R. Average over both leading dims.
            a0 = a.mean(axis=0)  # reduce d1
            return a0.mean(axis=0)  # reduce d2 -> (d3, d4)
        else:
            # use the only slice in dim0, mean over dim1 (folds)
            return a[0].mean(axis=0)
    elif a.ndim == 3 and a.shape[0] == 4:
        # (4, 16352, 89) -> mean over axis 0
        return a.mean(axis=0)
    elif a.ndim == 2:
        # Already (16352, 89)
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
            # ((a - clip_soft)^2) + clip_soft for masked entries only
            x_masked = x[mask] - clip_soft
            x_masked = x_masked * x_masked + clip_soft
            x[mask] = x_masked

        # inverse of sqrt-like transform
        x = np.power(x, 1.0 / float(track_transform))

    # log1p always applied in the R code after (re)scaling
    x = np.log1p(x)
    return x


def _build_exon_mask(gtf_df: pd.DataFrame,
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
    start = center_pos - seq_len // 2
    end = center_pos + seq_len // 2  # exclusive end

    # Filter GTF to exon rows, chromosome, and any overlap with [start, end)
    gtf = gtf_df
    # Expect columns: chr, annotation, seq, start1, end1, dot1, strand, dot2, gene
    gtf = gtf[(gtf["seq"] == "exon") &
              (gtf["chr"] == chrom) &
              (gtf["end1"] >= max(0, start)) &
              (gtf["start1"] <= end)]

    if gene_name is not None and len(gene_name) > 0:
        # R extracts gene symbols from the attribute column and matches exactly
        gtf = gtf[gtf["gene"] == gene_name]

    # If nothing matched, use default
    if gtf.shape[0] == 0:
        base = np.zeros(16384, dtype=np.float64)
        base[7209:7209 + 1966] = 1.0
        return base[16:16368]

    # Collect all exon base positions (can be large; we’ll convert to bins efficiently)
    # We can rasterize by bins instead of expanding every base.
    # Convert exon intervals to 0-based bin indices within the window.
    bin_idxs = []

    # Precompute bin starts in genomic coordinates relative to window start
    # We'll mark a bin as 1 if any base of that bin intersects an exon.
    for _, row in gtf.iterrows():
        s = int(row["start1"])
        e = int(row["end1"])  # inclusive in GTF; treat as inclusive for parity
        # Intersect with window [start, end)
        s_clip = max(s, start)
        e_clip = min(e, end - 1)
        if s_clip > e_clip:
            continue
        # Convert to positions relative to window start
        rel_s = s_clip - start
        rel_e = e_clip - start
        # Bin by floor(pos / 32)
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

    # R does additional frequency -> /32 -> thresholding; the effect is a "bin touched?"
    # The simpler union operation above matches that practical outcome for mask ∈ {0,1}.

    # Trim 16 bins each side to match (16352, 89) matrices
    return base[16:16368]


def _read_gtf(gtf_path: str) -> pd.DataFrame:
    """
    Read and lightly normalize the GTF into the column schema used in the R code:
      chr, annotation, seq, start1, end1, dot1, strand, dot2, gene
    The 'gene' column is parsed to a symbol before the dot (to mirror the R regexes).
    """
    # GTF has 9 columns; we keep them and parse attributes to gene_id/gene_name-ish behavior
    colnames = ["chr", "annotation", "seq", "start1", "end1", "dot1", "strand", "dot2", "gene"]
    df = pd.read_csv(gtf_path, sep="\t", header=None, comment="#", names=colnames, dtype=str)

    # Coerce numeric
    df["start1"] = df["start1"].astype(int)
    df["end1"] = df["end1"].astype(int)

    # The R code strips attributes down to a gene symbol before the first '.'
    # Assuming the attribute field has something like: gene_id "XYZ.1"; gene_name "XYZ"; ...
    # We'll first try to extract gene_name "..."; if missing, fall back to first quoted token.
    def parse_gene(attr: str) -> str:
        s = str(attr)
        # try gene_name "..."
        if 'gene_name "' in s:
            try:
                left = s.split('gene_name "', 1)[1]
                token = left.split('"', 1)[0]
                return token.split(".")[0]
            except Exception:
                pass
        # fallback: first quoted token
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
    # Remove trailing _wt or _mut
    if stem.endswith("_wt"):
        stem = stem[:-3]
    elif stem.endswith("_mut"):
        stem = stem[:-4]
    parts = stem.split("_")
    # Accept both 2-part (chr_pos) and 4-part (chr_pos_ref_alt)
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
                           gtf_df: pd.DataFrame,
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

    # Load arrays and rescale
    wt_raw = load_pickle(wt_path)
    mut_raw = load_pickle(mut_path)

    wt_scaled = _apply_rescale(wt_raw, rescale_tracks, scale_parameter, clip_soft, track_transform)
    mut_scaled = _apply_rescale(mut_raw, rescale_tracks, scale_parameter, clip_soft, track_transform)

    # Collapse to (16352, 89)
    wt_mat = _collapse_mean(wt_scaled)
    mut_mat = _collapse_mean(mut_scaled)

    if wt_mat.shape[0] != 16352 or wt_mat.shape[1] != 89:
        raise ValueError(f"{base}: unexpected collapsed shape {wt_mat.shape}, expected (16352, 89)")

    # Build exon mask
    mask = _build_exon_mask(gtf_df, chrom, center_pos, SEQ_LEN, default_gene)

    # Apply mask across rows
    wt_masked = wt_mat * mask[:, None]
    mut_masked = mut_mat * mask[:, None]

    # Column sums
    wt_sums = wt_masked.sum(axis=0)
    mut_sums = mut_masked.sum(axis=0)

    delta = mut_sums - wt_sums  # (89,)
    # Subset tracks
    delta_sub = delta[np.array(track_indices_0based, dtype=int)]

    # Row ID: drop suffixes; keep chr_pos or chr_pos_ref_alt if available
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
        "-o", "--output-csv", required=True,
        help="Path to write CSV. Rows = variants, columns = selected tracks."
    )
    parser.add_argument(
        "--gtf-path", default=GTF_PATH,
        help="Path to GTF. This argument is ignored if you would like to hard code it above."
    )
    parser.add_argument(
        "--gene", default=None,
        help="Optional constant gene symbol to use for all variants (e.g., 'BRCA1'). "
             "If omitted, mask falls back to the default center window as no exons will match."
    )
    parser.add_argument(
        "--gene-map", default=None,
        help="Optional CSV mapping each file 'row_id' (chr_pos[_ref_alt]) to a 'gene' column. "
             "Header must include: row_id,gene. Overrides --gene if provided."
    )
    parser.add_argument(
        "--no-rescale", default = RESCALE_TRACKS, action="store_true",
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

    # Load GTF (once)
    if not os.path.exists(args.gtf_path):
        sys.exit(f"GTF not found at {args.gtf_path}. Update GTF_PATH in the script or pass --gtf-path.")

    gtf_df = _read_gtf(args.gtf_path)

    # Discover pairs
    pairs = find_pairs(args.input_dir)
    if not pairs:
        sys.exit(f"No *_wt.obj with matching *_mut.obj found in {args.input_dir}")

    # Optional gene map CSV
    gene_map = {}
    if args.gene_map:
        gm = pd.read_csv(args.gene_map, header=None, names=["row_id", "gene"])
        if "row_id" not in gm.columns or "gene" not in gm.columns:
            sys.exit("gene-map CSV must have columns: row_id,gene")
        # normalize keys exactly as our row_id (chr_pos[_ref_alt])
        for _, r in gm.iterrows():
            rid = str(r["row_id"])
            gsym = str(r["gene"])
            if gsym and gsym.lower() != "nan":
                gene_map[rid] = gsym

    rows = []
    colnames = [f"track_{i+1}" for i in track_ix]

    # Process each pair
    for wt_path, mut_path in pairs:
        try:
            row_id, vec = compute_delta_for_pair(
                wt_path=wt_path,
                mut_path=mut_path,
                track_indices_0based=track_ix,
                gtf_df=gtf_df,
                default_gene=None,  # decide per-record below
                rescale_tracks=rescale_tracks,
                scale_parameter=float(args.scale_parameter),
                clip_soft=clip_soft,
                track_transform=float(args.track_transform),
            )

            # Determine gene to use for the mask (priority: gene-map > --gene > None)
            # Recompute mask+delta if we have a specific gene to apply.
            # We did one pass above with default_gene=None to get row_id; if a gene is specified,
            # recompute using that gene for exact parity with the R script's per-gene behavior.
            gene_for_this = gene_map.get(row_id, args.gene)
            if gene_for_this:
                # Re-do the computation with the explicit gene
                chrom, center_pos, _, _ = _extract_tokens_from_basename(os.path.basename(wt_path))
                wt_raw = load_pickle(wt_path)
                mut_raw = load_pickle(mut_path)
                wt_scaled = _apply_rescale(wt_raw, rescale_tracks, float(args.scale_parameter), clip_soft, float(args.track_transform))
                mut_scaled = _apply_rescale(mut_raw, rescale_tracks, float(args.scale_parameter), clip_soft, float(args.track_transform))
                wt_mat = _collapse_mean(wt_scaled)
                mut_mat = _collapse_mean(mut_scaled)
                mask = _build_exon_mask(gtf_df, chrom, center_pos, SEQ_LEN, gene_for_this)
                delta_full = (mut_mat * mask[:, None]).sum(axis=0) - (wt_mat * mask[:, None]).sum(axis=0)
                vec = delta_full[np.array(track_ix, dtype=int)]

            rows.append((row_id, *vec.tolist()))
        except Exception as e:
            print(f"[WARN] Skipping {wt_path} / {mut_path}: {e}", file=sys.stderr)

    if not rows:
        sys.exit("No rows produced; nothing to write.")

    # Write CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id"] + colnames)
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
