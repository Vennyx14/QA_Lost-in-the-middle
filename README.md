Pipeline cũ:
```
QA_Advanced_RAG/
├── data/
│   ├── raw/                 # Chứa file train.json, dev.json nguyên bản
│   ├── processed/           # Chứa file đã tách từ bằng underthesea và trộn nhiễu
│   └── index/               # Nơi lưu file database vector(VD:faiss_index.bin)
│ 
├── benchmark_tokenize
│ 
├── word2vec
│ 
├── src/
│   ├── config.py            # Chứa đường dẫn file và tên model
│   ├── __init__.py
│   ├── data_prep/           # Phụ trách xử lý văn bản tiếng Việt
│   │   ├── __init__.py
│   │   ├── segmenter.py
│   │   └── dataset_builder.py
│   ├── retriever/           # Phụ trách DPR và FAISS
│   │   ├── __init__.py
│   │   ├── embedding.py
│   │   └── faiss_index.py
│   ├── rl_reranker/         # Phụ trách Học tăng cường
│   │   ├── __init__.py
│   │   ├── environment.py   # Định nghĩa State, Action, Reward
│   │   └── ppo_agent.py     # Thuật toán tối ưu hóa policy
│   └── evaluation/          # Phụ trách đo lường 
│       ├── __init__.py
│       └── metrics.py
├── notebooks/               # Nơi chạy nháp, test thuật toán lẻ
├── test_retriever.py        # Test truy xuất dữ liệu
├── main.py                  # File khởi chạy toàn bộ luồng
├── requirements.txt         # underthesea, sentence-transformers, faiss-cpu, torch
└── README.md
```

Pipeline New:
```
QA_Advanced_RAG/
├── data/
│   ├── raw/                # File train.json, dev.json nguyên bản
│   ├── processed/          # Data sau khi qua VnCoreNLP (Word Segmented)
│   ├── index/              # faiss_index.bin, doc_map.pkl
│   └── samples/            # Dataset cho huấn luyện PhoBERT (Query-Sentence pairs)
│
├── src/
│   ├── config.py           # Quản lý TOKEN_LIMIT=256, paths, model names
│   ├── __init__.py
│   │
│   ├── data_prep/          # Xử lý văn bản (Thành viên 1 & 2)
│   │   ├── __init__.py
│   │   ├── segmenter.py    # VnCoreNLP Word Segmentation
│   │   └── splitter.py     # Sentence Splitter (Tách câu để nén)
│   │
│   ├── retriever/          # Phụ trách Bi-Encoder & FAISS
│   │   ├── __init__.py
│   │   ├── embedding.py    # BKAI Bi-encoder (768-dim)
│   │   ├── faiss_index.py  # IndexFlatIP & L2 Normalization
│   │   └── search.py       # Wrapper tìm kiếm Top-K docs
│   │
│   ├── compressor/         # TRỌNG TÂM: Semantic Context Compression
│   │   ├── __init__.py
│   │   ├── model.py        # Định nghĩa PhoBERT Cross-Encoder
│   │   ├── scorer.py       # Tính điểm tương quan Query-Sentence
│   │   ├── selector.py     # Chiến thuật chọn câu (Ranker & 256-token limit,etc)
│   │   └── train_scorer.py # Script fine-tune PhoBERT
│   │
│   └── evaluation/         # Đo lường & Baseline
│       ├── __init__.py
│       ├── metrics.py      # MRR
│       └── benchmark.py    # So sánh Baseline (Top-1) vs Compression (PhoBERT)
│
├── notebooks/              # Phân tích dữ liệu, test thử nghiệm nén
├── scripts/                # Các script chạy một lần (VD: build_index.py)
├── main.py                 # Khởi chạy toàn bộ luồng: Retrieve -> Compress -> Evaluate
├── requirements.txt        # py-vncorenlp, sentence-transformers, transformers, faiss-cpu,...
└── README.md
```

1. Thay thế rl_reranker bằng compressor
-  Module này không chỉ xếp hạng lại (rerank) mà còn thực hiện nhiệm vụ Abstraction (Trích xuất đặc tính).

-  scorer.py: Sử dụng PhoBERT làm Cross-Encoder. Thay vì đưa vào cả văn bản,ta đưa vào từng câu.

-  selector.py: Nó sẽ nhận điểm từ scorer, sắp xếp câu, và nhặt các câu có điểm cao nhất cho đến khi chạm mốc 256 tokens. Ở phần này sẽ thử các phương án khác nhau để cho ra output.

2. Nâng cấp data_prep với splitter.py
-  Vì mục tiêu là nén ngữ cảnh theo cấp độ câu,cần một bộ tách câu chuẩn. VnCoreNLP hỗ trợ việc này rất tốt. Việc tách câu phải diễn ra trước khi nạp vào compressor.

3. Module evaluation tập trung vào "The Lift"
-  Báo cáo khoa học cần số liệu chứng minh. Cần sử dụng benchmark.py để chạy hai luồng song song:

-  Luồng A (Baseline): Lấy Top-1 văn bản từ FAISS.

-  Luồng B (Proposed): Lấy Top-20, tách câu, nén bằng PhoBERT.

-> Kết quả: So sánh MRR của Luồng B so với Luồng A.
