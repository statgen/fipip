#!/usr/bin/env python3
"""
Build a master dataframe from Enformer .h5 files.
"""

import argparse
import os
import re
import sys
from typing import Dict, List, Sequence, Tuple, Set, Optional

import h5py
import numpy as np
import pandas as pd

def _decode_str_array(arr: Sequence) -> List[str]:
    """Decode a NumPy array of bytes/objects into a list of Python str."""
    out: List[str] = []
    for x in arr:
        if isinstance(x, (bytes, bytearray)):
            out.append(x.decode("utf-8", errors="ignore"))
        else:
            out.append(str(x))
    return out


def _normalize_chr(c: str) -> str:
    c = str(c)
    return c if c.startswith("chr") else f"chr{c}"


def _variant_key(chr_s: str, pos: int | str, ref: str, alt: str) -> str:
    return f"{_normalize_chr(chr_s)}_{int(pos)}_{ref}_{alt}"


def _parse_variant_key(key: str) -> Optional[Tuple[str, int, str, str]]:
    """
    Parse 'chr_pos_ref_alt' (or '1_pos_ref_alt') into (chrX, pos:int, ref, alt).
    Returns None if format doesn't match 4-part underscore-separated token.
    """
    if not key:
        return None
    parts = key.split("_")
    if len(parts) != 4:
        return None
    chrom, pos, ref, alt = parts
    chrom = _normalize_chr(chrom)
    try:
        pos = int(pos)
    except Exception:
        return None
    return chrom, pos, ref, alt


CHR_TOKEN_RE = re.compile(r'^(?:chr)?(\d{1,2})$', flags=re.IGNORECASE)

def _variant_chr_to_int(variant_key: str) -> Optional[int]:
    """
    Extract chromosome number (1..22) from a variant key like 'chr1_123_A_G' or '1_123_A_G'.
    Returns None if it can't parse or it's outside 1..22.
    """
    if not variant_key:
        return None
    chrom_token = variant_key.split('_', 1)[0]
    m = CHR_TOKEN_RE.match(chrom_token)
    if not m:
        return None
    n = int(m.group(1))
    return n if 1 <= n <= 22 else None


def load_list_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

def find_h5_map(h5_dir: str) -> Dict[int, str]:
    """
    Return a mapping {chrom: path} for chroms 1..22 found in the directory.
    Preference:
      1) exact legacy: 1000G.MAF_threshold__0.005.{i}.h5
      2) any .h5 whose name clearly contains that chromosome number (robust fallback)
    """
    mapping: Dict[int, str] = {}

    # 1) Exact legacy names first (your case)
    for i in range(1, 23):
        p = os.path.join(h5_dir, f"1000G.MAF_threshold__0.005.{i}.h5")
        if os.path.exists(p):
            mapping[i] = p

    # 2) Flexible scan for any others
    def _extract_chr_num(fname: str) -> Optional[int]:
        m = re.search(r"chr(\d{1,2})(?!\d)", fname, flags=re.IGNORECASE)
        if m:
            n = int(m.group(1))
            return n if 1 <= n <= 22 else None
        m = re.search(r"(^|[^\d])(\d{1,2})([^\d]|$)", fname)
        if m:
            n = int(m.group(2))
            return n if 1 <= n <= 22 else None
        return None

    for fname in os.listdir(h5_dir):
        if not fname.lower().endswith(".h5"):
            continue
        n = _extract_chr_num(fname)
        if n is None:
            continue
        path = os.path.join(h5_dir, fname)
        if (n not in mapping) or (os.path.basename(path) < os.path.basename(mapping[n])):
            mapping[n] = path

    return mapping


def read_targets_from_h5(h5_path: str) -> List[str]:
    with h5py.File(h5_path, "r") as f:
        if "target_labels" not in f:
            raise KeyError(f"target_labels not found in {h5_path}")
        labels = f["target_labels"][:]
    return _decode_str_array(labels)


def intersect_targets(h5_targets: Sequence[str], requested: Sequence[str]) -> List[str]:
    """Return targets in H5 order, filtered by requested. If requested is empty, return all."""
    if not requested:
        return list(h5_targets)
    h5_set = {t.strip() for t in h5_targets}
    req_set = {t.strip() for t in requested if t.strip()}
    inter = [t for t in h5_targets if t in req_set]  # preserve H5 order
    missing = sorted(req_set - h5_set)
    if missing:
        sys.stderr.write(f"[warning] {len(missing)} requested targets not in H5; ignoring.\n")
    if not inter:
        raise ValueError("None of the requested targets are present in the H5 files.")
    return inter

def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate Enformer SAD across H5s into a single table.")
    ap.add_argument("--h5-dir", required=True, help="Directory containing chromosome .h5 files (1-22).")
    ap.add_argument("--targets-file", required=False, help="Optional file: one target per line. If omitted, all targets are used.")
    ap.add_argument("--variants-file", required=False, help="Optional file of variants to keep.")
    ap.add_argument("--output", required=True, help="Output file path.")
    args = ap.parse_args()

    h5_dir = os.path.abspath(args.h5_dir)
    if not os.path.isdir(h5_dir):
        ap.error(f"--h5-dir not found or not a directory: {h5_dir}")

    # Discover H5s
    h5_map = find_h5_map(h5_dir)
    if not h5_map:
        raise SystemExit("No H5 files found in the directory.")
    missing_all = [i for i in range(1, 23) if i not in h5_map]
    if missing_all:
        sys.stderr.write(f"[warning] Some chromosomes missing H5s: {missing_all}\n")

    # Load variants-of-interest (optional), normalize chr token, and collect desired chromosomes
    variant_order: List[str] = []
    variant_keep: Set[str] = set()
    desired_chroms: Optional[List[int]] = None

    if args.variants_file:
        raw_lines = load_list_file(args.variants_file)

        # Support formats: single token 'chr_pos_ref_alt' OR 4 columns: chr pos ref alt
        def parse_variant_line(line: str) -> Optional[str]:
            s = line.strip()
            if not s or s.startswith('#'):
                return None
            parts = re.split(r"[\t, \s]+", s)
            if len(parts) == 1:
                token = parts[0]
                # Normalize chr if it's like "1_123_A_G"
                if token.startswith("chr"):
                    return token
                else:
                    tok_parts = token.split("_")
                    if len(tok_parts) == 4 and tok_parts[0].isdigit():
                        return _variant_key(f"chr{tok_parts[0]}", tok_parts[1], tok_parts[2], tok_parts[3])
                    return token
            if len(parts) >= 4:
                c, p, r, a = parts[:4]
                return _variant_key(c, p, r, a)
            return None

        for raw in raw_lines:
            key = parse_variant_line(raw)
            if key and key not in variant_keep:
                variant_keep.add(key)
                variant_order.append(key)

        if variant_order:
            sys.stderr.write(f"[info] Loaded {len(variant_order)} requested variants.\n")
            chroms = { _variant_chr_to_int(v) for v in variant_order }
            chroms.discard(None)
            desired_chroms = sorted(c for c in chroms if 1 <= c <= 22)
            if not desired_chroms:
                sys.stderr.write("[warning] --variants-file parsed, but no valid chromosomes found (1..22). Processing all.\n")

    # Choose a file to read target labels from: first desired chrom if set, else smallest chrom we have
    first_chrom = desired_chroms[0] if desired_chroms else min(h5_map.keys())
    first_file = h5_map[first_chrom]
    h5_targets = read_targets_from_h5(first_file)

    # Targets selection
    requested_targets = load_list_file(args.targets_file) if args.targets_file else []
    selected_targets = intersect_targets(h5_targets, requested_targets)

    # Build target indices in H5 order
    index_by_name = {name: idx for idx, name in enumerate(h5_targets)}
    target_indices = np.asarray([index_by_name[t] for t in selected_targets], dtype=np.int64)

    # h5py: strictly increasing + dedup for the slice, then map back to original order
    uniq_idx = np.unique(target_indices)                      # sorted by value
    sorted_idx = np.sort(uniq_idx).astype(np.int64)
    cols_reorder = np.searchsorted(sorted_idx, target_indices).astype(np.int64)
    if len(uniq_idx) != len(target_indices):
        sys.stderr.write("[warning] Duplicate targets requested; duplicates will be collapsed in the HDF5 read.\n")
    sorted_idx_list = sorted_idx.tolist()  # for maximum h5py compatibility

    # Iterate chromosomes
    iter_chroms = desired_chroms if desired_chroms else sorted(h5_map.keys())

    # Optional warning for requested chroms that are missing H5s
    if desired_chroms:
        missing = [c for c in desired_chroms if c not in h5_map]
        if missing:
            sys.stderr.write(f"[warning] No H5 found for requested chromosomes: {missing}\n")

    variant_ids: List[str] = []
    per_chr_rows: List[np.ndarray] = []

    for chrom in iter_chroms:
        h5_path = h5_map.get(chrom)
        if not h5_path:
            continue

        with h5py.File(h5_path, "r") as f:
            # Basic structure checks
            if "SAD" not in f:
                raise KeyError(f"SAD dataset not found in {h5_path}")

            sad_ds = f["SAD"]
            n0, n1 = sad_ds.shape

            # Read targets in sorted-index order, then restore original order
            if n0 == len(h5_targets):
                # SAD is (targets x variants)
                sad_sorted = sad_ds[sorted_idx_list, :]             # (len(unique_targets) x variants)
                sad_sel    = sad_sorted.T[:, cols_reorder]          # -> (variants x selected_targets)
            elif n1 == len(h5_targets):
                # SAD is (variants x targets)
                sad_sorted = sad_ds[:, sorted_idx_list]             # (variants x len(unique_targets))
                sad_sel    = sad_sorted[:, cols_reorder]            # -> (variants x selected_targets)
            else:
                raise ValueError(
                    f"SAD shape {sad_ds.shape} does not match number of targets {len(h5_targets)} in {h5_path}"
                )

            # Build descriptor lists (normalized) from large universe
            chr_full = [_normalize_chr(x) for x in _decode_str_array(f["chr"][:])]
            pos_full = [int(p) for p in f["pos"][:]]
            ref_full = _decode_str_array(f["ref"][:])
            alt_full = _decode_str_array(f["alt"][:])

            # Map SAD columns to descriptor indices or IDs if available
            local_variants: Optional[List[str]] = None
            for key in ("variant_ids", "variant_indices", "variant_index", "var_idx", "indices"):
                if key in f:
                    if key == "variant_ids":
                        local_variants = _decode_str_array(f[key][:])   # one ID per SAD column
                    else:
                        idx = np.asarray(f[key][:]).astype(int).ravel()
                        local_variants = [
                            _variant_key(chr_full[i], pos_full[i], ref_full[i], alt_full[i])
                            for i in idx
                        ]
                    break

            # Fallback: assume 1:1 alignment between SAD columns and descriptor arrays
            if local_variants is None:
                local_variants = [
                    _variant_key(c, p, r, a) for c, p, r, a in zip(chr_full, pos_full, ref_full, alt_full)
                ]

            # Make sure sizes line up
            if sad_sel.shape[0] != len(local_variants):
                raise ValueError(
                    f"Dimension mismatch in {h5_path}: SAD has {sad_sel.shape[0]} variants, "
                    f"but descriptors map has {len(local_variants)}"
                )

            # --- Allele-aware filtering ---
            if variant_keep:
                # Build fast lookup maps for exact and swapped keys -> row index in sad_sel
                exact_to_idx: Dict[str, int] = {v: i for i, v in enumerate(local_variants)}

                def swapped_key(key: str) -> Optional[str]:
                    parsed = _parse_variant_key(key)
                    if not parsed:
                        return None
                    c, p, r, a = parsed
                    return _variant_key(c, p, a, r)  # swap ref/alt

                # Process only requested variants for this chromosome, in their original order
                chrom_requests = [v for v in variant_order if _variant_chr_to_int(v) == chrom and v in variant_keep]
                kept_rows: List[np.ndarray] = []
                kept_ids: List[str] = []

                for req in chrom_requests:
                    if req in exact_to_idx:
                        kept_rows.append(sad_sel[exact_to_idx[req], :])
                        kept_ids.append(req)
                    else:
                        sw = swapped_key(req)
                        if sw and sw in exact_to_idx:
                            # allele-swapped match: multiply by -1
                            kept_rows.append(-1.0 * sad_sel[exact_to_idx[sw], :])
                            kept_ids.append(req)  # keep original requested orientation in output
                        # else: not found in this H5; skip

                if kept_rows:
                    per_chr_rows.append(np.vstack(kept_rows))
                    variant_ids.extend(kept_ids)
            else:
                # No variant filter: keep all in file order
                variant_ids.extend(local_variants)
                per_chr_rows.append(sad_sel)

        sys.stderr.write(f"[info] processed chr {chrom}: {h5_path}\n")

    if not per_chr_rows:
        raise SystemExit("No rows collected (check inputs and filters).")

    # Stack and build DataFrame
    all_matrix = np.vstack(per_chr_rows)
    df = pd.DataFrame(all_matrix, columns=selected_targets)
    df.insert(0, "hg19_variant", variant_ids)

    # If variants were provided, reorder rows to match the input order and warn on missing
    if variant_order:
        present = set(df["hg19_variant"])  # actually found (exact or swapped)
        missing = [v for v in variant_order if v not in present]
        if missing:
            sys.stderr.write(f"[warning] {len(missing)} requested variants not found in any H5 and were omitted.\n")
        df = df.set_index("hg19_variant").reindex([v for v in variant_order if v in present]).reset_index()

    # Write output
    out = args.output
    ext = os.path.splitext(out)[1].lower()
    if ext in (".csv", ".tsv"):
        sep = "," if ext == ".csv" else "\t"
        df.to_csv(out, sep=sep, index=False)
    else:
        sys.stderr.write(f"[warning] Unrecognized extension {ext!r}; writing CSV.\n")
        df.to_csv(out, index=False)

    print(f"Wrote {len(df):,} variants x {len(selected_targets)} targets -> {out}")

if __name__ == "__main__":
    main()
