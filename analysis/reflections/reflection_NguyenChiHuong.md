# Individual Reflection — Nguyễn Chí Hương

## Phần 1: Mapping bài giảng vào implementation

| Lecture concept | Module | Hàm/class | Observation thực tế |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | Với corpus hiện tại tạo 208 chunks, trung bình 99 ký tự; basic tạo 51 chunks, trung bình 410 ký tự. Threshold cao tạo ranh giới semantic chi tiết hơn. |
| Parent-child retrieval | M1 | `chunk_hierarchical()` | Tạo 106 child chunks từ 26 tài liệu trong pipeline; child phục vụ precision, parent giữ context để trả lời câu nhiều ý. |
| BM25 + Dense fusion | M2 | `HybridSearch`, `reciprocal_rank_fusion()` | BM25 xử lý exact term/số, BGE-M3 xử lý semantic; RRF hợp nhất rank mà không cần chuẩn hóa hai loại score. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Reranker sắp lại candidate theo cặp query-document; 5/5 unit tests pass, đổi lại latency CPU cao hơn retrieval thuần túy. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Wrapper có fallback và giữ đủ 20 record khi evaluator không khả dụng. Run này chưa có điểm thật vì API key trống. |
| Contextual enrichment | M5 | `_enrich_single_call()`, `enrich_chunks()` | Combined mode giới hạn một API call/chunk; local fallback đã enrich 106/106 chunks trong lần chạy không-key. |

## Phần 2: Khó khăn và cách giải quyết

### 1. Qdrant chưa hoạt động

- **Exact error:** `WinError 10061: No connection could be made because the target machine actively refused it`.
- **Debug:** Traceback dừng tại `recreate_collection()` tới `localhost:6333`.
- **Giải quyết:** Khởi động Docker Desktop, chạy `docker compose up -d` và xác nhận container Qdrant ở trạng thái `Up`.

### 2. Docker engine chưa sẵn sàng

- **Exact error:** `open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified`.
- **Debug:** Docker CLI có mặt nhưng named pipe của Linux Engine chưa tồn tại.
- **Giải quyết:** Mở Docker Desktop, chờ engine sẵn sàng rồi chạy lại Compose.

### 3. API hết credit

- **Exact error:** `429 ... insufficient_quota ... credit_balance_exhausted`.
- **Debug:** Các RAGAS jobs đều retry nhưng không thể thành công do billing, không phải rate limit tạm thời.
- **Giải quyết:** Xóa key không còn quota; bổ sung nhánh skip/fallback để pipeline chạy local và ghi rõ metric `0.0` là “chưa đo”.

### 4. Encoding Windows trong pytest

- **Exact error:** `UnicodeEncodeError: 'charmap' codec can't encode characters`.
- **Debug:** Lỗi xuất hiện khi pytest capture dòng cảnh báo có emoji dưới code page `cp1252`, không nằm trong logic chunking/evaluation.
- **Giải quyết:** Đổi warning trong module sang ASCII và dùng `PYTHONIOENCODING=utf-8` khi chạy pipeline.

### 5. Thời gian inference

- Lượt test tổng đầu tiên timeout sau 120 giây vì tải/chạy model nặng.
- Tách test theo module để khoanh vùng; sau khi cache model, xác nhận tổng cộng 37/37 test pass.
- Pipeline đầy đủ mất 746,8 giây, trong đó dense indexing production mất 169 giây.

## Phần 3: Action Plan cho project

### Project: Trợ lý tra cứu chính sách nội bộ

#### Hiện tại

- Pipeline: Markdown/PDF → hierarchical chunking → enrichment → BM25 + dense Qdrant → RRF → cross-encoder → grounded answer → RAGAS.
- Known issues: tài liệu nhiều phiên bản, CPU latency cao, PDF scan chưa OCR và chưa có evaluator API để đo RAGAS thật.

#### Plan áp dụng

1. [ ] **Chunking:** Dùng hierarchical làm mặc định; structure-aware cho tài liệu có bảng/header và OCR trước với PDF scan.
2. [ ] **Search:** Giữ hybrid BM25 + dense vì corpus có cả từ khóa chính xác, số tiền và câu hỏi diễn đạt tự nhiên.
3. [ ] **Versioning:** Thêm metadata `version`, `effective_date`, `status`; filter bản đã thay thế trước khi fusion/rerank.
4. [ ] **Reranking:** Giữ BGE reranker cho top-20 → top-3; benchmark FlashRank nếu SLA thấp hơn 1 giây.
5. [ ] **Evaluation:** Chạy RAGAS với key còn quota, thêm exact-match cho numeric/version/negation và lưu latency theo stage.
6. [ ] **Enrichment:** Dùng combined single-call trong production; cache kết quả theo hash của chunk để không gọi lại khi dữ liệu không đổi.

#### Timeline

- **Tuần 1:** OCR, chuẩn hóa metadata phiên bản và thêm version-aware filters.
- **Tuần 2:** Batch/caching embedding, benchmark BGE-M3 và model nhẹ hơn.
- **Tuần 3:** Chạy RAGAS thật, phân tích bottom-5 và bổ sung regression test.
- **Tuần 4:** Theo dõi latency/cost, hoàn thiện dashboard và quy trình cập nhật tài liệu.
