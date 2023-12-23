MODELLER Averaging for Protein Structure Prediction

This repository contains a script for utilizing averaging methods in MODELLER for protein structure prediction. The script is designed to enhance the accuracy and reliability of predicted protein structures through model averaging.
Authors

Command-line Arguments

The script can be executed from the command line with the following options:

    --alnfile: Path to the alignment file (e.g., 'tar_tem_alignment.ali').
    --knowns: Target structure name (e.g., '1bdm').
    --sequence: Label (e.g., 'TvLDH').

Optional Arguments

    --n_jobs (default: 4): Number of parallel jobs to use during model averaging.
    --n_runs (default: 24): Number of MD runs to perform.
    --model_name (default: 'TvLDH_averaged'): Name of the output model file.
    --scoring_func (default: 'dope'): Scoring function for model evaluation.
    --select_top (default: 0.1): Fraction of top decoys to be used for averaging.

Example Usage

    python altmod_run_averaging.py --alnfile tar_tem_alignment.ali --knowns 1bdm --sequence TvLDH

Notes

- The script initializes an 'automodel' object using the Automodel_averaging class.
- Additional parameters, such as md_level, can be adjusted directly within the script if needed.
- The final averaged model will be written to the specified output file.
- If you want to build multiple averaged models for the same protein, remember to change the random seed of MODELLER within the script, by adding the lines:

    random_seed = 42
    env.io.random_seed = random_seed

