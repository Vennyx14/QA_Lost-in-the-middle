# Mẫu src/compressor/scorer.py
import numpy as np

class DummyScorer:
    """Dùng để test Pipeline trước khi có PhoBERT thật của Trung"""
    def __init__(self):
        pass
        
    def score_sentences(self, query, sentences):
        # Giả lập trả về mảng điểm số ngẫu nhiên từ 0 đến 1
        # Kích thước mảng bằng đúng số lượng câu
        return np.random.rand(len(sentences))