
"""P4: tối ưu siêu tham số XGBoost bằng Optuna với TPE sampler."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data import load_processed
from src.tune import run_tuning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--trials",
        type=int,
        default=config.N_TRIALS,
        help=f"Số cấu hình thử nghiệm (mặc định: {config.N_TRIALS}).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
    )

    config.ensure_dirs()

    X_train, _, y_train, _ = load_processed()

    print(
        f"P4 - Optuna/TPE | Train: {len(y_train):,} dòng | "
        f"Gian lận: {int(y_train.sum()):,} | "
        f"Trials: {args.trials} | CV: {config.N_FOLDS}-Fold"
    )

    best = run_tuning(
        X_train,
        y_train,
        n_trials=args.trials,
    )

    print("\nKẾT QUẢ TUNING")
    print(f"PR-AUC CV tốt nhất: {best['best_pr_auc_cv']:.6f}")
    print(f"Trial tốt nhất: {best['best_trial_number']}")
    print(f"Phương pháp mất cân bằng: {best['imbalance']}")

    print("\nSiêu tham số tốt nhất:")
    for name, value in best["params"].items():
        print(f"  {name}: {value}")

    print("\nCÁC FILE KẾT QUẢ P4")
    print(f"1. Best parameters: {config.BEST_PARAMS_PATH}")
    print(f"2. Trial history: {config.METRICS_DIR / 'tuning_trials.csv'}")
    print(f"3. Model comparison: {config.METRICS_DIR / 'tuning_compare.csv'}")
    print(
        "4. Optimization history: "
        f"{config.FIGURES_DIR / '06_optimization_history.png'}"
    )


if __name__ == "__main__":
    main()
