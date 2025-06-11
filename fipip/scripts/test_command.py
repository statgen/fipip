import sys, os, gzip, argparse, logging, warnings, shutil, subprocess, ast, math
import pandas as pd
import numpy as np

from fipip.utils.utils import flexopen

def parse_arguments(_args):
    repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    parser = argparse.ArgumentParser(prog=f"fipip test-command", description="Test command")

    inout_params = parser.add_argument_group("Input/Output Parameters", "Input/output directory/files.")
    inout_params.add_argument('--input', type=str, required=True, help='Input file')

    if len(_args) == 0:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args(_args)


def test_command(_args):

    # parse argument
    args = parse_arguments(_args)
    
    # identify the sample name from the VCF file
    print(f"Input file: {args.input}")


if __name__ == "__main__":
    # Get the base file name without extension
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    # Dynamically get the function based on the script name
    func = getattr(sys.modules[__name__], script_name)

    # Call the function with command line arguments
    func(sys.argv[1:])
