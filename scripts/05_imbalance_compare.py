"""05_imbalance_compare.py: so sánh 4 cách xử lý mất cân bằng với XGBoost.

Đọc : data/processed/train.parquet
Ghi : results/metrics/imbalance_compare.csv, results/predictions/xgb_<biến thể>_oof.csv,
      results/params/chosen_imbalance.json, figures/05_*.png
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.compare import choose_imbalance, compare_variants, plot_variant_comparison
from src.data import load_processed
from src.evaluate import plot_pr_curves, save_metrics

logger = logging.getLogger(__name__)


def main() -> None:
    """Chạy so sánh 4 biến thể xử lý mất cân bằng trên tập train bằng 5-Fold CV."""
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
    config.ensure_dirs()

    X_train, _, y_train, _ = load_processed()
    logger.info(
        "Train: %d dòng, %d ca gian lận (%.3f%%)",
        len(y_train),
        int(y_train.sum()),
        100 * float(y_train.mean()),
    )

    df, oofs = compare_variants(X_train, y_train)
    metrics_path = save_metrics(df, "imbalance_compare")
    for variant, oof in oofs.items():
        pd.DataFrame({"y_true": y_train.values, "y_proba": oof}).to_csv(
            config.PREDICTIONS_DIR / f"xgb_{variant}_oof.csv",
            index=False,
        )

    compare_fig = plot_variant_comparison(
        df, config.FIGURES_DIR / "05_imbalance_compare.png"
    )
    pr_fig = plot_pr_curves(
        {variant: (y_train, oof) for variant, oof in oofs.items()},
        config.FIGURES_DIR / "05_pr_curves.png",
    )

    choice = choose_imbalance(df)
    config.CHOSEN_IMBALANCE_PATH.write_text(
        json.dumps(choice, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Chọn: %s | %s", choice["imbalance"], choice["reason"])

    print("\nSo sánh các kỹ thuật xử lý mất cân bằng (5-Fold CV trên tập train):")
    print(df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nPhương pháp được chọn: {choice['imbalance']}")
    print(f"Lý do: {choice['reason']}")
    print(f"Đã lưu bảng chỉ số: {metrics_path}")
    print(f"Đã lưu lựa chọn: {config.CHOSEN_IMBALANCE_PATH}")
    print(f"Đã lưu biểu đồ: {compare_fig}, {pr_fig}")


if __name__ == "__main__":
    main()
