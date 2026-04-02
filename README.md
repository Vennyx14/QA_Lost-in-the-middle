```
Pipeline:
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
│   ├── retriever/           # Phụ trách DPR và FAISS (Thành viên 1)
│   │   ├── __init__.py
│   │   ├── embedding.py
│   │   └── faiss_index.py
│   ├── rl_reranker/         # Phụ trách Học tăng cường (Thành viên 2)
│   │   ├── __init__.py
│   │   ├── environment.py   # Định nghĩa State, Action, Reward
│   │   └── ppo_agent.py     # Thuật toán tối ưu hóa policy
│   └── evaluation/          # Phụ trách đo lường (Thành viên 3)
│       ├── __init__.py
│       └── metrics.py
├── notebooks/               # Nơi chạy nháp, test thuật toán lẻ
├── test_retriever.py        # Test truy xuất dữ liệu
├── main.py                  # File khởi chạy toàn bộ luồng
├── requirements.txt         # underthesea, sentence-transformers, faiss-cpu, torch
└── README.md
```
