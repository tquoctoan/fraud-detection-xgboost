# Chương 5: Phân tích khám phá dữ liệu (EDA)

EDA tổng quan được thực hiện trên toàn bộ dữ liệu đã làm sạch sau khi ghép hai
tập train và test. Các thống kê chỉ phục vụ mô tả dữ liệu; không dùng test để
huấn luyện mô hình. Bảng mô tả được lưu tại `results/metrics/eda_describe.csv`,
bảng tần suất và tỷ lệ nhãn tại `results/metrics/eda_label_counts.csv`.

## 5.1. Thống kê mô tả và phân bố nhãn

Sau khi loại các dòng trùng, dữ liệu có 283.726 giao dịch, trong đó 283.253
giao dịch bình thường và 473 giao dịch gian lận. Lớp gian lận chiếm khoảng
0,167% tổng số quan sát, thể hiện sự mất cân bằng nghiêm trọng. Biểu đồ
`figures/02_label_distribution.png` dùng trục số giao dịch dạng log để thể hiện
rõ cả lớp đa số và lớp thiểu số.

## 5.2. Phân tích Amount

`Amount` có phân bố lệch phải; biểu đồ histogram và boxplot sử dụng
`log(1 + Amount)` để thể hiện được các giao dịch giá trị thấp lẫn phần đuôi
phân bố mà không làm mất các giá trị bằng 0. Đây chỉ là biến đổi để trực quan
hóa, không thay đổi dữ liệu Parquet. Các biểu đồ được lưu lần lượt tại
`figures/02_amount_hist.png` và `figures/02_amount_box.png`.

## 5.3. Phân tích Time

`Time` biểu thị số giây kể từ giao dịch đầu tiên. Biểu đồ
`figures/02_time_dist.png` tổng hợp số giao dịch và tỷ lệ gian lận theo từng
khoảng một giờ (`Time // 3600`), qua đó thể hiện biến động hoạt động và gian
lận theo thời gian.

Phân tích chi tiết các biến PCA `V1`–`V28` và tương quan được trình bày ở các
mục tiếp theo.
