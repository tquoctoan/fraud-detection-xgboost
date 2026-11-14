# Kết luận và Hướng phát triển

## 1. Tổng kết kết quả đạt được
Trong phạm vi bài tập lớn và đặc biệt là nhiệm vụ P3 (So sánh các kỹ thuật xử lý dữ liệu mất cân bằng), nghiên cứu đã đạt được các kết quả nổi bật sau:
1. **Khảo sát toàn diện bài toán:** Làm rõ mức độ mất cân bằng cực đoan (0.167% gian lận, tương ứng tỷ lệ 1:599) và phân tích sâu sắc cơ chế gây thiên lệch của hàm mất mát truyền thống cùng nghịch lý độ chính xác (Accuracy Paradox).
2. **Xây dựng hệ thống thực nghiệm chuẩn mực:** Thiết kế pipeline chống rò rỉ dữ liệu (No Data Leakage) nghiêm ngặt với Stratified 5-Fold Cross-Validation và đánh giá độc lập trên tập Test gồm 56,746 giao dịch gốc.
3. **Thực nghiệm so sánh đa chiều 4 phương pháp:**
   - **Baseline:** Xác lập chuẩn đối chứng với Accuracy 99.95%, Precision 94.59%, Recall 73.68%, F1-score 0.8284.
   - **Random Undersampling:** Đạt Recall cao nhất (87.37%) nhưng gây ra sự sụp đổ thảm khốc về Precision (chỉ còn 5.84%) với 1,339 ca báo động giả do mất mát 99.8% dữ liệu lớp đa số.
   - **SMOTE:** Cải thiện Recall lên 80.00% nhưng làm giảm Precision xuống 87.36% (11 ca báo sai) và tốn thêm 28% thời gian huấn luyện.
   - **Class Weight:** Đạt hiệu năng toàn diện vượt trội nhất với F1-score = 0.8721, PR-AUC = 0.8225, Precision = 0.9740 và Recall = 0.7895 (chỉ 2 ca báo sai).
4. **Bàn giao kết quả thống nhất:** Lựa chọn phương pháp **Class Weight (`scale_pos_weight = 599.48`)** làm phương pháp xử lý mất cân bằng tối ưu và lưu vào `results/params/chosen_imbalance.json` để phục vụ bước tinh chỉnh siêu tham số tiếp theo (P4).

## 2. Hạn chế của nghiên cứu
- Các kỹ thuật Resampling chỉ mới khảo sát trên các biến thể cơ bản (SMOTE chuẩn và Random Undersampling), chưa mở rộng sang các biến thể lai nâng cao như Borderline-SMOTE, SMOTE-ENN, hay EasyEnsemble.
- Việc đánh giá so sánh ở P3 đang cố định ngưỡng phân loại tại giá trị mặc định 0.5. Việc tinh chỉnh ngưỡng động (Threshold Tuning) sẽ được thực hiện chuyên sâu ở bước P2 (mục 6.6).

## 3. Hướng phát triển trong tương lai
- Thử nghiệm các kiến trúc Deep Learning chuyên dụng cho dữ liệu dạng chuỗi thời gian hoặc Graph Neural Networks (GNN) để nắm bắt mối quan hệ mạng lưới giữa các tài khoản gian lận.
- Kết hợp kỹ thuật Semi-Supervised Learning hoặc Anomaly Detection (Isolation Forest, Autoencoders) đối với các trường hợp gian lận mới phát sinh chưa từng có trong tập huấn luyện (Zero-day Fraud).
