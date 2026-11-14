"""05_imbalance_compare.py — Thực nghiệm và so sánh các kỹ thuật xử lý dữ liệu mất cân bằng (P3).

Triển khai 4 trường hợp:
1. Baseline (Không xử lý)
2. Class Weight (scale_pos_weight)
3. SMOTE (Synthetic Minority Over-sampling Technique)
4. Random Undersampling (RandomUnderSampler)

Đầu vào:
  - data/processed/train.parquet
  - data/processed/test.parquet

Đầu ra:
  - results/metrics/imbalance_compare.csv (Kết quả 5-Fold Cross-Validation)
  - results/metrics/imbalance_test_compare.csv (Kết quả trên tập Test độc lập)
  - results/predictions/xgb_<biến thể>_oof.csv
  - results/predictions/xgb_<biến thể>_test.csv
  - results/params/chosen_imbalance.json
  - figures/05_imbalance_compare.png
  - figures/05_pr_curves.png
  - figures/05_f1_precision_recall.png
  - figures/05_confusion_matrices.png
  - figures/05_resampling_distributions.png
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Đảm bảo import được gói src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config
from src.compare import (
    choose_imbalance,
    compare_variants,
    evaluate_variants_on_test,
    plot_all_confusion_matrices,
    plot_f1_precision_recall_comparison,
    plot_resampling_distributions,
    plot_variant_comparison,
)
from src.data import load_processed
from src.evaluate import get_classification_report_str, plot_pr_curves, save_metrics

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
log = logging.getLogger(__name__)


def main() -> None:
    config.ensure_dirs()
    log.info("========== BẮT ĐẦU THỰC NGHIỆM P3: SO SÁNH XỬ LÝ MẤT CÂN BẰNG ==========")

    # 1. Đọc dữ liệu đã qua tiền xử lý chuẩn
    X_train, X_test, y_train, y_test = load_processed()
    n_train = len(y_train)
    fraud_train = int(y_train.sum())
    n_test = len(y_test)
    fraud_test = int(y_test.sum())

    log.info(
        "Dữ liệu huấn luyện: %d mẫu (%d ca gian lận, tỷ lệ %.4f%%)",
        n_train,
        fraud_train,
        (fraud_train / n_train) * 100,
    )
    log.info(
        "Dữ liệu kiểm thử:    %d mẫu (%d ca gian lận, tỷ lệ %.4f%%)",
        n_test,
        fraud_test,
        (fraud_test / n_test) * 100,
    )

    # 2. Vẽ biểu đồ phân phối dữ liệu trước và sau khi áp dụng từng kỹ thuật
    log.info("Vẽ biểu đồ phân phối số lượng mẫu trước và sau xử lý mất cân bằng...")
    dist_fig_path = config.FIGURES_DIR / "05_resampling_distributions.png"
    plot_resampling_distributions(X_train, y_train, dist_fig_path)
    log.info("Đã lưu biểu đồ phân phối mẫu tại: %s", dist_fig_path)

    # 3. Thực nghiệm 1: Đánh giá Cross-Validation (5-Fold CV trên tập Train)
    log.info("Thực hiện 5-Fold Stratified Cross-Validation trên tập Train...")
    df_cv, oofs = compare_variants(X_train, y_train, config.IMBALANCE_CHOICES)
    save_metrics(df_cv, "imbalance_compare")

    # Lưu dự đoán out-of-fold cho từng biến thể
    for v, oof in oofs.items():
        oof_path = config.PREDICTIONS_DIR / f"xgb_{v}_oof.csv"
        pd.DataFrame({"y_true": y_train.values, "y_proba": oof}).to_csv(
            oof_path, index=False
        )
    log.info("Đã lưu các file dự đoán OOF tại: %s", config.PREDICTIONS_DIR)

    # 4. Thực nghiệm 2: Đánh giá trực tiếp trên tập Test độc lập (giữ nguyên phân phối Test)
    log.info("Đánh giá mô hình của 4 phương pháp trên tập Test độc lập (ngưỡng 0.5)...")
    df_test, test_probas, fitted_models = evaluate_variants_on_test(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        variants=config.IMBALANCE_CHOICES,
        threshold=config.DEFAULT_THRESHOLD,
    )
    save_metrics(df_test, "imbalance_test_compare")

    # Lưu dự đoán tập Test
    for v, proba in test_probas.items():
        test_pred_path = config.PREDICTIONS_DIR / f"xgb_{v}_test.csv"
        pd.DataFrame({"y_true": y_test.values, "y_proba": proba}).to_csv(
            test_pred_path, index=False
        )

    # In Classification Report chi tiết cho từng phương pháp trên tập Test
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT TRÊN TẬP TEST GỐC CỦA CÁC PHƯƠNG PHÁP:")
    print("=" * 70)
    for v in config.IMBALANCE_CHOICES:
        proba = test_probas[v]
        y_pred = (proba >= config.DEFAULT_THRESHOLD).astype(int)
        report_str = get_classification_report_str(y_test, y_pred)
        print(f"\n--- Phương pháp: {df_test.loc[df_test['imbalance_key'] == v, 'Phương pháp'].values[0]} ({v}) ---")
        print(report_str)

    # In Bảng so sánh kết quả trên tập Test (Đúng theo yêu cầu đề bài Phần 4)
    table_cols = [
        "Phương pháp",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "Macro F1",
        "Weighted F1",
        "PR-AUC",
    ]
    df_display = df_test[table_cols].copy()
    print("\n" + "=" * 70)
    print("BẢNG SO SÁNH KẾT QUẢ TRÊN TẬP TEST (NGƯỠNG = 0.5):")
    print("=" * 70)
    print(df_display.to_markdown(index=False, floatfmt=".4f"))
    print("=" * 70 + "\n")

    # 5. Vẽ các biểu đồ so sánh kết quả
    log.info("Đang tạo các biểu đồ so sánh và lưu vào thư mục figures/...")
    
    # a. So sánh PR-AUC từ Cross-Validation
    plot_variant_comparison(df_cv, config.FIGURES_DIR / "05_imbalance_compare.png")

    # b. So sánh đường cong Precision-Recall trên out-of-fold
    plot_pr_curves(
        {v: (y_train, o) for v, o in oofs.items()},
        config.FIGURES_DIR / "05_pr_curves.png",
    )

    # c. So sánh F1, Precision và Recall trên tập Test
    plot_f1_precision_recall_comparison(
        df_test, config.FIGURES_DIR / "05_f1_precision_recall.png"
    )

    # d. So sánh Ma trận nhầm lẫn (Confusion Matrix) của 4 phương pháp
    plot_all_confusion_matrices(
        y_test,
        test_probas,
        config.DEFAULT_THRESHOLD,
        config.FIGURES_DIR / "05_confusion_matrices.png",
    )

    # 6. Chọn phương pháp tối ưu và lưu vào chosen_imbalance.json
    choice = choose_imbalance(df_cv)
    config.CHOSEN_IMBALANCE_PATH.write_text(
        json.dumps(choice, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("KẾT LUẬN LỰA CHỌN PHƯƠNG PHÁP:")
    log.info("Phương pháp được chọn: %s", choice["imbalance"])
    log.info("Lý do lựa chọn: %s", choice["reason"])
    log.info("========== HOÀN THÀNH THỰC NGHIỆM P3 ==========")


if __name__ == "__main__":
    main()
