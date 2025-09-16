#!/usr/bin/env python3
"""
alphagenome_cli.py

Score variants using AlphaGenome matching per-gene rows when possible, and write results to CSV.

Input format (TSV by default, no header):
    chr1_115746_C_T    ENSG00000238009
    chr1_135203_G_A    ENSG00000238009

Examples:
  # basic usage (API key via env)
  ALPHAGENOME_API_KEY=sk_... python alphagenome_cli.py \
      -i ./example_data.txt \
      -o alphagenome_results.csv

  # custom delimiter
  ALPHAGENOME_API_KEY=sk_... python alphagenome_cli.py \
      -i variants.tsv --sep "\t" -o out.csv
"""

from __future__ import annotations
from pathlib import Path
import argparse
import os
import time
import sys
import pandas as pd

from alphagenome.data import genome
from alphagenome.models import dna_client
from alphagenome.models import variant_scorers

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Score variants with AlphaGenome RNA-seq tracks scores."
    )
    p.add_argument(
        "-i", "--input", required=True,
        help="Path to an input file with two columns (no header): variant, gene. "
             "Variant format must be CHR_POS_REF_ALT."
    )
    p.add_argument(
        "-o", "--output", default="alphagenome_results.csv",
        help="Output CSV filename (default: alphagenome_results.csv)."
    )
    p.add_argument(
        "--sep", default="\t",
        help=r"Input delimiter (default: '\t')."
    )
    p.add_argument(
        "--checkpoint-every", type=int, default=5000,
        help="Write a checkpoint CSV every N processed variants (default: 5000)."
    )
    p.add_argument(
        "--sleep", type=float, default=1.0,
        help="Seconds to sleep between variants (default: 1.0)."
    )
    p.add_argument(
        "--api-key",
        help="AlphaGenome API key; if omitted, reads from ALPHAGENOME_API_KEY env var."
    )
    return p.parse_args()


def get_api_key(cli_value: str | None) -> str:
    key = cli_value or os.environ.get("ALPHAGENOME_API_KEY")
    if not key:
        raise RuntimeError(
            "No API key provided. Use --api-key or set ALPHAGENOME_API_KEY in the environment."
        )
    return key


def read_input(path: str, sep: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    df = pd.read_csv(p, sep=sep, header=None, names=["variant", "gene"], dtype=str)
    df["variant"] = df["variant"].str.strip()
    df["gene"] = df["gene"].str.strip()
    df = df.dropna(subset=["variant", "gene"])
    return df

def parse_variant(variant_str: str) -> genome.Variant:
    """
    Accept 'CHR_POS_REF_ALT' where CHR may be 'chr1'..'chr22','chrX','chrY'.
    Examples: 'chr1_12345_A_T'
    """
    chrom, pos, ref, alt = variant_str.split("_")
    return genome.Variant(
        chromosome=chrom,
        position=int(pos),
        reference_bases=ref,
        alternate_bases=alt,
    )


def main():
    args = parse_args()
    try:
        api_key = get_api_key(args.api_key)
    except Exception as e:
        print(f"[fatal] {e}", file=sys.stderr)
        sys.exit(2)
    
    # Initialize model client
    dna_model = dna_client.create(api_key)

    # Read inputs
    try:
        df_input = read_input(args.input, args.sep)
    except Exception as e:
        print(f"[fatal] Failed to read input: {e}", file=sys.stderr)
        sys.exit(2)

    results = []
    track_names = None
    processed = 0

    for _, row in df_input.iterrows():
        processed += 1
        variant_str = row["variant"]
        gene_id_target = str(row["gene"]).split(".", 1)[0]  # strip version if present

        try:
            variant = parse_variant(variant_str)
            interval = variant.reference_interval.resize(dna_client.SEQUENCE_LENGTH_1MB)

            # Score variant with recommended RNA-SEQ scorers
            variant_scorer = variant_scorers.RECOMMENDED_VARIANT_SCORERS["RNA_SEQ"] # Feel free to change this if RNA-seq is not desired
            scored = dna_model.score_variant(
                interval=interval, variant=variant, variant_scorers=[variant_scorer]
            )

            if track_names is None:
                try:
                    track_names = list(scored[0].var["name"])
                except Exception:
                    track_names = [f"track_{i}" for i in range(scored[0].X.shape[1])]

            obs = scored[0].obs.copy()
            obs_gene_ids = obs["gene_id"].astype(str).str.split(".", n=1).str[0]
            matched = obs_gene_ids == gene_id_target

            if matched.any():
                row_idx = obs.index.get_loc(obs.index[matched.argmax()])
                score_vec = scored[0].X[row_idx, :].flatten()
                fallback = 0
            else:
                score_vec = scored[0].X.mean(axis=0).flatten()
                fallback = 1

            results.append([variant_str, gene_id_target, fallback] + list(score_vec))

        except Exception as e:
            print(f"[warn] Skipping {variant_str} / {gene_id_target}: {e}", file=sys.stderr)

        if args.checkpoint_every and processed % args.checkpoint_every == 0:
            if track_names and results:
                cols = ["variant", "gene", "fallback"] + track_names
                pd.DataFrame(results, columns=cols).to_csv(args.output, index=False)
                print(f"[info] checkpoint @ {processed} -> {args.output}")

        if args.sleep and args.sleep > 0:
            time.sleep(args.sleep)

    if not track_names:
        print("[error] No successful scores—nothing to write.", file=sys.stderr)
        sys.exit(1)

    cols = ["variant", "gene", "fallback"] + track_names
    pd.DataFrame(results, columns=cols).to_csv(args.output, index=False)
    print(f"[done] Wrote {len(results)} rows to: {args.output}")


if __name__ == "__main__":
    main()
