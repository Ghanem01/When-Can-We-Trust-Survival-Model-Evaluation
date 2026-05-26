# run_experiment.py
# ============================================================
# Generic launcher for all survival models (CoxPH, Weibull, RSF, NMTLR, DeepHit)
# Usage:
#   python run_experiment.py --dataset metabric --model rsf --censorship 30 --n_trials 10 --n_outer 5 --n_inner 3
# ============================================================

import argparse
from all_models import run_model
from load_data import load_dataset

# -----------------------------
# Argument parsing
# -----------------------------
parser = argparse.ArgumentParser(description="Run nested CV experiment for a survival model")
parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., metabric, nacd, flchain, ...)')
parser.add_argument('--model', type=str, required=True, help='Model name (coxph, weibull, rsf, nmtlr, deephit)')
parser.add_argument('--n_trials', type=int, default=10, help='Number of Optuna trials for hyperparameter tuning')
parser.add_argument('--n_outer', type=int, default=5, help='Number of outer folds')
parser.add_argument('--n_inner', type=int, default=3, help='Number of inner folds')
parser.add_argument('--censorship', type=int, required=True, help='Censorship rate (e.g., 10, 30, 50)')
parser.add_argument('--replications', type=int, default=5, help='Number of replications per configuration')
parser.add_argument('--force_test_censoring', action='store_true', help='Force high test censoring (~90%) for outer test folds')
args = parser.parse_args()

# -----------------------------
# Main experiment loop
# -----------------------------
for repl in range(1, args.replications + 1):
    print(f"\n=== Dataset: {args.dataset} | Model: {args.model.upper()} | Censorship: {args.censorship}% | Replication: {repl} ===")

    df, features_col = load_dataset(args.dataset, args.censorship, repl)

    df_std, df_true = run_model(
        model_name=args.model,
        df=df,
        features_col=features_col,
        n_splits_outer=args.n_outer,
        n_splits_inner=args.n_inner,
        n_trials=args.n_trials,
        censorship=args.censorship,
        replication=repl,
        dataset_name=args.dataset,
        force_test_censoring=args.force_test_censoring
    )

    print(f" Finished replication {repl} | std shape={df_std.shape}, true shape={df_true.shape}")
