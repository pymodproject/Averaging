import os
import argparse
from modeller import *
from modeller.automodel import *
from altmod_averaging import Automodel_averaging, run_averaging

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script description here.')
    parser.add_argument('--alnfile', type=str, required=True,
                        help='Path to the alignment file')
    parser.add_argument('--knowns', type=str, required=True,
                        help='Known structures (comma-separated)')
    parser.add_argument('--sequence', type=str, required=True,
                        help='Target sequence')
    parser.add_argument('--n_jobs', type=int, default=4,
                        help='Number of parallel jobs')
    parser.add_argument('--n_runs', type=int, default=24,
                        help='Number of MD runs')
    parser.add_argument('--model_name', type=str, default='TvLDH_averaged',
                        help='Name of the output model')
    parser.add_argument('--scoring_func', type=str, default='dope',
                        help='Scoring function for model evaluation')
    parser.add_argument('--select_top', type=float, default=0.1,
                        help='Select top fraction of decoys for averaging')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    # Set up the environment.
    example_dirpath = os.path.dirname(__file__)
    env = environ()
    env.io.atom_files_directory.append(example_dirpath)

    # Initialize an 'automodel' object using the 'Automodel_averaging' class.
    a = Automodel_averaging(env, alnfile=args.alnfile,
                            knowns=args.knowns.split(","), sequence=args.sequence)
    
    # Set other parameters based on command-line arguments
    a.md_level = refine.slow
    run_averaging(a, n_jobs=args.n_jobs, n_runs=args.n_runs,
                  model_name=args.model_name, scoring_func=args.scoring_func,
                  select_top=args.select_top)
