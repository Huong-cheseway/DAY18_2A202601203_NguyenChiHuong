# Báo cáo hoàn thành — Lab 18: Production RAG

**Sinh viên:** Nguyễn Chí Hướng

**Ngày:** 18/08/2026

**Hình thức:** Bài cá nhân

## Module và kiểm thử

| Module | Nội dung | Kết quả |
|---|---|---:|
| M1 | Semantic, hierarchical, structure-aware chunking | 13/13 test pass |
| M2 | Vietnamese BM25, dense Qdrant, RRF | 5/5 test pass |
| M3 | CrossEncoder và FlashRank reranking | 5/5 test pass |
| M4 | RAGAS wrapper và failure analysis | 4/4 test pass |
| M5 | Summary, HyQA, contextual prepend, metadata, combined mode | 10/10 test pass |
| **Tổng** | | **37/37 test pass** |

## Kết quả chạy pipeline

- Qdrant chạy thành công bằng Docker tại `localhost:6333`.
- Baseline tạo 57 basic paragraph chunks.
- Production tạo 106 hierarchical child chunks từ 26 tài liệu.
- M5 enrich đủ 106/106 chunks bằng local fallback do không có API key.
- Pipeline hoàn tất 20/20 câu hỏi trong 746,8 giây trên CPU.
- Hai report được tạo tại `reports/naive_baseline_report.json` và `reports/ragas_report.json`.

## Kết quả RAGAS

| Metric | Naive | Production | Δ | Ghi chú |
|---|---:|---:|---:|---|
| Faithfulness | 0.0000 | 0.0000 | 0.0000 | Chưa đo, evaluator API không được cấu hình |
| Answer Relevancy | 0.0000 | 0.0000 | 0.0000 | Chưa đo, evaluator API không được cấu hình |
| Context Precision | 0.0000 | 0.0000 | 0.0000 | Chưa đo, evaluator API không được cấu hình |
| Context Recall | 0.0000 | 0.0000 | 0.0000 | Chưa đo, evaluator API không được cấu hình |

Các số 0 là fallback kỹ thuật để pipeline/report có cấu trúc hợp lệ, không phải phép đo chất
lượng. Không đưa ra kết luận production tốt hơn baseline khi chưa có evaluator.

## Key Findings

1. **Điểm mạnh:** Pipeline local hoàn chỉnh từ load PDF/Markdown đến chunk, enrich, hybrid search và rerank; toàn bộ 37 test pass.
2. **Thách thức lớn nhất:** Inference BGE-M3 trên CPU chậm; indexing production mất 169 giây và toàn pipeline mất 746,8 giây.
3. **Rủi ro retrieval:** Corpus cố ý chứa chính sách cũ/mới rất giống nhau, cần metadata versioning thay vì chỉ dựa vào semantic score.
4. **Giới hạn đánh giá:** Không có OpenAI credit nên chưa thu được RAGAS thật; failure analysis hiện dùng phân tích thủ công có ghi nhãn rõ ràng.

