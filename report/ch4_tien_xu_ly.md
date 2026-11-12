# Chương 4: Tiền xử lý dữ liệu

## 4.1. Nguồn và cấu trúc dữ liệu

Dữ liệu giao dịch thẻ tín dụng được lấy từ OpenML, Dataset ID 1597; nếu đã có
`data/raw/creditcard.csv`, quy trình sử dụng bản CSV này. Tập dữ liệu đầu vào
phải có 284.807 dòng và đủ 31 cột: `Time`, `V1`–`V28`, `Amount` và `Class`.
`Time` được giữ như một biến đặc trưng, không bị loại như mã định danh.

## 4.2. Kiểm tra và làm sạch

Nhãn `Class` được chuyển sang kiểu số nguyên và kiểm tra chỉ gồm hai giá trị
0 (giao dịch bình thường) và 1 (gian lận). Quy trình kiểm tra giá trị thiếu
trên toàn bộ 31 cột; nếu phát hiện giá trị thiếu thì dừng với thông báo cụ thể,
không âm thầm loại bỏ hoặc điền giá trị. Các bản ghi trùng được xác định trên
đầy đủ 31 cột và giữ lại lần xuất hiện đầu tiên. Dữ liệu chuẩn không có giá trị
thiếu; có 1.081 dòng trùng được loại bỏ, còn 283.726 dòng, gồm 473 trường hợp
gian lận.

Không chuẩn hóa đặc trưng và không áp dụng SMOTE ở bước làm sạch. Việc xử lý
mất cân bằng, nếu cần, thuộc các bước xây dựng mô hình sau này.

## 4.3. Chia tập và lưu trữ

Dữ liệu được chia thành tập huấn luyện và kiểm thử theo tỷ lệ 80/20, dùng
`stratify=Class` và seed 42 để bảo toàn tỷ lệ nhãn và tái lập kết quả. Tập train
có 226.980 dòng (378 gian lận); tập test có 56.746 dòng (95 gian lận). Hai tập
được lưu dưới dạng Parquet tại `data/processed/train.parquet` và
`data/processed/test.parquet`, mỗi tập gồm 30 biến đặc trưng và cột `Class`.

Tóm tắt số dòng, số bản ghi trùng và số ca gian lận được lưu tại
`results/metrics/preprocess_summary.csv`.
