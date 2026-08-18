# Failure Analysis — Lab 18: Production RAG

**Sinh viên:** Nguyễn Chí Hướng

**Ngày chạy:** 18/08/2026

**Test set:** 20 câu hỏi

## Lưu ý về phép đo

Pipeline baseline và production đều đã chạy hết 20 câu hỏi. Tuy nhiên `OPENAI_API_KEY`
không được cấu hình nên RAGAS được bỏ qua có chủ đích; bốn giá trị `0.0` dưới đây là
giá trị fallback, không phải kết luận rằng chất lượng hệ thống bằng 0. Vì không có điểm
per-question thật, năm trường hợp dưới đây là phân tích rủi ro thủ công trên các câu khó
đại diện, không được trình bày như “bottom-5 theo RAGAS”.

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ | Trạng thái |
|---|---:|---:|---:|---|
| Faithfulness | 0.0000 | 0.0000 | 0.0000 | Chưa đo — thiếu evaluator API |
| Answer Relevancy | 0.0000 | 0.0000 | 0.0000 | Chưa đo — thiếu evaluator API |
| Context Precision | 0.0000 | 0.0000 | 0.0000 | Chưa đo — thiếu evaluator API |
| Context Recall | 0.0000 | 0.0000 | 0.0000 | Chưa đo — thiếu evaluator API |

## Năm trường hợp rủi ro cao

### 1. Chính sách phép năm có nhiều phiên bản

- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** 15 ngày theo chính sách v2024; bản v2023 ghi 12 ngày và đã bị thay thế.
- **Observed:** Report fallback không lưu câu trả lời per-question nên chưa có output để chấm.
- **Error Tree:** Output chưa đo → corpus có cả bản cũ và mới → retrieval có thể lấy nhầm phiên bản.
- **Root cause tiềm năng:** Dense/BM25 coi hai văn bản gần như cùng chủ đề; pipeline chưa filter theo `version`, `effective_date` hay trạng thái hiện hành.
- **Suggested fix:** Trích xuất metadata phiên bản, ưu tiên ngày hiệu lực mới nhất và loại tài liệu có trạng thái `ĐÃ THAY THẾ` trước rerank.

### 2. Chu kỳ đổi mật khẩu có xung đột phiên bản

- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** 120 ngày theo v2.0; 90 ngày là chính sách v1.0 cũ.
- **Observed:** Chưa có điểm/output RAGAS per-question.
- **Error Tree:** Output chưa đo → context có thể chứa 90 và 120 ngày → thiếu bước resolution theo phiên bản.
- **Root cause tiềm năng:** Semantic similarity rất cao giữa `mat_khau_v1.md` và `mat_khau_v2.md`.
- **Suggested fix:** Boost tài liệu có nhãn “hiện hành”, penalize “đã thay thế”, và yêu cầu answer nêu rõ phiên bản/ngày hiệu lực.

### 3. Câu hỏi multi-hop về phép và lương

- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 18 ngày phép và 20–35 triệu VNĐ/tháng.
- **Observed:** Chưa có điểm/output RAGAS per-question.
- **Error Tree:** Output chưa đo → cần context từ hai tài liệu → top-k hoặc reranker có thể chỉ giữ một nguồn.
- **Root cause tiềm năng:** `RERANK_TOP_K=3` có thể thiếu một trong hai mảnh bằng chứng; generation fallback chỉ trả context đầu tiên.
- **Suggested fix:** Decompose query thành hai sub-query, retrieve theo từng ý rồi hợp nhất context có citation nguồn.

### 4. Mua laptop cần kết hợp ba điều kiện

- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Director phê duyệt, phòng CNTT xác nhận cấu hình và phải có ít nhất ba báo giá.
- **Observed:** Chưa có điểm/output RAGAS per-question.
- **Error Tree:** Output chưa đo → đúng tài liệu nhưng nhiều đoạn/bảng/list → chunk con có thể không giữ đủ ba quy tắc.
- **Root cause tiềm năng:** Hierarchical child retrieval có độ chính xác cao nhưng câu trả lời cần toàn bộ parent context.
- **Suggested fix:** Khi child khớp, trả parent tương ứng cho generator; giữ nguyên bảng và danh sách bằng structure-aware chunking.

### 5. Tạm ứng quá hạn cần tính toán số học

- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày; 2%/tháng là 300.000 đồng/tháng, pro-rata khoảng 50.000 đồng cho 5 ngày.
- **Observed:** Chưa có điểm/output RAGAS per-question.
- **Error Tree:** Context có quy tắc → cần suy luận ngày quá hạn và pro-rata → generator/context-only fallback không bảo đảm phép tính.
- **Root cause tiềm năng:** Retrieval giải quyết bằng chứng nhưng không tự thực hiện phép tính đáng tin cậy.
- **Suggested fix:** Thêm calculator/tool cho câu numeric và unit test công thức `amount × monthly_rate × overdue_days / 30`.

## Case Study trình bày

Chọn câu hỏi phép năm v2024 vì nó thể hiện lỗi phổ biến của production RAG: retrieval có thể
tìm đúng chủ đề nhưng sai hiệu lực văn bản. Error Tree: kiểm tra output → kiểm tra context có
cả 12/15 ngày → kiểm tra metadata phiên bản → ưu tiên v2024 và loại v2023 đã bị thay thế.

Nếu có thêm một giờ, ưu tiên bổ sung metadata `version`, `effective_date`, `status`, sau đó
chạy lại RAGAS với API key còn quota để thay các phân tích proxy bằng bottom-5 có số đo thật.
