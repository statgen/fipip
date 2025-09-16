#!/usr/bin/env python3
# borzoi_1.py

import os
import argparse
import json

import pickle

import numpy as np
import pandas as pd
import pysam

import h5py
import tensorflow as tf
import baskerville
from baskerville import seqnn
from baskerville import gene as bgene

import pyfaidx

from borzoi_helpers import *  # KEEP: uses your existing process_sequence & predict_tracks


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Borzoi predictions for variants provided as chr_pos_ref_alt (e.g., chr1_123_A_T), no header."
    )
    p.add_argument("-i", "--input", required=True, help="Path to input file with one variant per line.")
    p.add_argument("-o", "--outdir", required=True, help="Directory to write output wild type and mutant .obj files. ~11 MB per file.")
    return p.parse_args()


def _parse_variant_token(variant_token):
    """
    Parse 'chr1_123456_A_T' into (chrom, pos:int, ref:str, alt:str).
    Expected format: <chrom>_<pos>_<ref>_<alt>, where ref/alt are A/C/G/T.
    """
    tok = str(variant_token).strip()
    if tok == "" or tok.lower() == "nan":
        raise ValueError("Empty line")
    parts = tok.split("_")
    if len(parts) != 4:
        raise ValueError(f"Variant must have 4 parts separated by '_': got {tok}")
    chrom, pos_s, ref, alt = parts
    try:
        pos = int(pos_s)
    except Exception:
        raise ValueError(f"Position must be an integer in variant '{tok}'")
    ref = ref.upper()
    alt = alt.upper()
    if ref not in ("A", "C", "G", "T") or alt not in ("A", "C", "G", "T"):
        raise ValueError(f"Ref/Alt must be A/C/G/T in variant '{tok}'")
    return chrom, pos, ref, alt


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # --------------------
    # Hard-coded model config
    # --------------------
    params_file = 'params_pred.json'
    targets_file = 'targets_gtex.txt'  # Subset of targets_human.txt

    seq_len = 524288
    n_folds = 4
    rc = True

    # --------------------
    # Read model parameters
    # --------------------
    with open(params_file) as params_open:
        params = json.load(params_open)
        params_model = params['model']
        params_train = params['train']

    # ---------------
    # Read targets
    # ---------------
    targets_df = pd.read_csv(targets_file, index_col=0, sep='\t')
    target_index = targets_df.index

    # Create local index of strand_pair (relative to sliced targets)
    if rc:
        strand_pair = targets_df.strand_pair
        target_slice_dict = {ix: i for i, ix in enumerate(target_index.values.tolist())}
        slice_pair = np.array(
            [target_slice_dict[ix] if ix in target_slice_dict else ix for ix in strand_pair.values.tolist()],
            dtype='int32'
        )

    # -------------------------
    # Initialize model ensemble
    # -------------------------
    models = []
    for fold_ix in range(n_folds):
        model_file = f"saved_models/f{fold_ix}/model0_best.h5"
        seqnn_model = seqnn.SeqNN(params_model)
        seqnn_model.restore(model_file, 0)
        seqnn_model.build_slice(target_index)
        if rc:
            seqnn_model.strand_pair.append(slice_pair)
        seqnn_model.build_ensemble(rc, [0])
        models.append(seqnn_model)

    # -----------------------------
    # Initialize fasta sequence I/O
    # -----------------------------
    fasta_open = pysam.Fastafile('hg38.fa')

    # -----------------------------
    # Load splice site annotation (parity with original; not used below)
    # -----------------------------
    try:
        _splice_df = pd.read_csv('gencode41_basic_protein_splice.csv.gz', sep='\t', compression='gzip')
    except Exception:
        _splice_df = None

    # -----------------------------------
    # Load variants file
    # Uses first/only column as the variant token; ignores blank lines
    # -----------------------------------
    with open (args.input) as f:
      variant_tokens = [line.strip() for line in f if line.strip()]

    # ----------------
    # Iterate variants
    # ----------------
    problematic = []

    for token in variant_tokens:
        try:
            chrom, center_pos, ref_base, alt_base = _parse_variant_token(token)
        except Exception as e:
            problematic.append(("PARSE", str(token), str(e)))
            continue
        
        transcriptome = bgene.Transcriptome('gencode41_basic_nort.gtf')

        poses = [center_pos]
        alts = [alt_base]
        ref = [ref_base]

        start = center_pos - seq_len // 2
        end = center_pos + seq_len // 2

        # Use first model for stride/length metadata
        seq_out_start = start + models[0].model_strides[0] * models[0].target_crops[0]
        seq_out_len   = models[0].model_strides[0] * models[0].target_lengths[0]

        # One-hot encode WT (helper kept intact)
        sequence_one_hot_wt = process_sequence(fasta_open, chrom, start, end)

        # Induce mutation(s)
        sequence_one_hot_mut = np.copy(sequence_one_hot_wt)

        for pos, alt in zip(poses, alts):
            if alt == 'A':
                alt_ix = 0
            elif alt == 'C':
                alt_ix = 1
            elif alt == 'G':
                alt_ix = 2
            elif alt == 'T':
                alt_ix = 3
            else:
                problematic.append((chrom, pos, ref, alt, "Unsupported ALT base"))
                continue

            arr_idx = pos - start - 1  # original indexing convention
            if arr_idx < 0 or arr_idx >= sequence_one_hot_mut.shape[0]:
                problematic.append((chrom, pos, ref, alt, "ALT position out of bounds"))
                continue

            sequence_one_hot_mut[arr_idx] = 0.0
            sequence_one_hot_mut[arr_idx, alt_ix] = 1.0

        # Make predictions (helpers from borzoi_helpers)
        y_wt = predict_tracks(models, sequence_one_hot_wt)
        y_mut = predict_tracks(models, sequence_one_hot_mut)

        # Compose output paths in requested directory
        base = f"{chrom}_{''.join([str(i) for i in poses])}_{''.join(ref)}_{''.join(alts)}"
        filename_wt = os.path.join(args.outdir, f"{base}_wt.obj")
        filename_mut = os.path.join(args.outdir, f"{base}_mut.obj")

        with open(filename_wt, 'wb') as fh_wt, open(filename_mut, 'wb') as fh_mut:
            pickle.dump(y_wt, fh_wt)
            pickle.dump(y_mut, fh_mut)

    if problematic:
        print("Completed with warnings on some records (showing up to 10):")
        for item in problematic[:10]:
            print("  ", item)


if __name__ == "__main__":
    main()
