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
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
    config.ensure_dirs()

    X_train, _, y_train, _ = load_processed()
    print(
        f"P4 - Optuna/TPE | Train: {len(y_train):,} dòng | "
        f"Gian lận: {int(y_train.sum()):,} | Trials: {args.trials} | "
        f"CV: {config.N_FOLDS}-Fold"
    )

    best = run_tuning(X_train, y_train, n_trials=args.trials)
    print("\nKẾT QUẢ TUNING")
    print(f"PR-AUC CV tốt nhất: {best['best_pr_auc_cv']:.6f}")
    print(f"Trial tốt nhất: {best['best_trial_number']}")
    print(f"Phương pháp mất cân bằng: {best['imbalance']}")
    print("Siêu tham số tốt nhất:")
    for name, value in best["params"].items():
        print(f"  {name}: {value}")
    print(f"\nĐã lưu: {config.BEST_PARAMS_PATH}")
    print(f"Đã lưu lịch sử: {config.METRICS_DIR / 'tuning_trials.csv'}")


if __name__ == "__main__":
    main()
