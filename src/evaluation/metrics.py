import math
import stat

import numpy as np
import re
from collections import Counter, defaultdict

class QAMetrics:
    """
    Lớp cung cấp các phương thức đo lường hiệu quả cho hệ thống RAG nén ngữ cảnh.
    """

    # @staticmethod
    # def normalize_text(text):
    #     """
    #     Chuẩn hóa văn bản: viết thường, xóa dấu câu và khoảng trắng thừa.
    #     Giúp việc so sánh giữa văn bản gốc và văn bản nén công bằng hơn.
    #     """
    #     text = text.lower()
    #     # Xóa các ký tự đặc biệt/dấu câu
    #     text = re.sub(r'[^\w\s_]', '', text) 
    #     # Xóa khoảng trắng thừa
    #     text = " ".join(text.split())
    #     return text
    
    # @staticmethod
    # def compression_stats(original_text_list, compressed_text):
    #     """
    #     Tính toán các chỉ số về nén.
    #     - compression_ratio: Tỷ lệ độ dài sau nén / trước nén (càng thấp nén càng nhiều).
    #     """
    #     original_full_text = " ".join(original_text_list)
    #     len_orig = len(original_full_text.split())
    #     len_comp = len(compressed_text.split())
        
    #     ratio = len_comp / len_orig if len_orig > 0 else 0
    #     return {
    #         "original_tokens": len_orig,
    #         "compressed_tokens": len_comp,
    #         "compression_ratio": round(ratio, 4)
    #     }

    @staticmethod
    def calculate_recall(k, scores, sentences, docs_id, ground_truth_id):
        """
        Recall: Trả về 1 nếu có câu trong tập đáp án đúng nằm trong top k
        """
        combined = list(zip(scores, docs_id))
        combined_sorted = sorted(combined, key=lambda x: x[0], reverse=True)
        top_k_docs = [item[1] for item in combined_sorted[:k]]
        if ground_truth_id in top_k_docs:
            return 1
        return 0

    @staticmethod
    def calculate_f1(predicted_context, ground_truth_answer):
        """
        F1-Score: Đo độ trùng lặp ở cấp độ token giữa đoạn nén và đáp án.
        Rất hữu ích khi đoạn nén chỉ giữ lại được một phần của đáp án.
        """
        pred_tokens = QAMetrics.normalize_text(predicted_context).split()
        truth_tokens = QAMetrics.normalize_text(ground_truth_answer).split()
        
        if not pred_tokens or not truth_tokens:
            return 0
        
        common = Counter(pred_tokens) & Counter(truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0
        
        precision = num_same / len(pred_tokens)
        recall = num_same / len(truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def calculate_mrr(relevance_list):
        """
        MRR (Mean Reciprocal Rank)
        Dùng để đánh giá vị trí của văn bản đúng sau khi Bicoder lấy ra các tập văn bản liên quan nhất.
        relevance_list: List các giá trị 0 và 1. 
        Ví dụ: [0, 1, 0] nghĩa là đáp án nằm ở vị trí thứ 2.
        """
        for i, score in enumerate(relevance_list):
            if score > 0:
                return 1 / (i + 1)
        return 0
    
    @staticmethod
    def calculate_log_discount(scores, sentences, docs_ids):
        combined = list(zip(scores, docs_ids))
        combined_sorted = sorted(combined, key=lambda x: x[0], reverse=True)
        # dict để chứa tổng điểm của mỗi tập văn bản
        doc_scores = defaultdict(float)

        for i, (_, docs_id) in enumerate(combined_sorted):
            rank = i+1
            weight = 1 / math.log2(rank + 1)
            doc_scores[docs_id] += weight
            
        # sắp xếp lại các văn bản theo tổng điểm tính được
        # output là list các tuple [(doc_id, score)]
        doc_final_scores = sorted(doc_scores.items(), key=lambda x:x[1], reverse = True)

        ranked_docs_id = [item[0] for item in doc_final_scores]
        ranked_score = [item[1] for item in doc_final_scores]
        # Trả về list docs id đã sắp xếp, và list điểm tương ứng
        return ranked_docs_id, ranked_score


# Ví dụ sử dụng
if __name__ == "__main__":
    k = 2
    scores = [0.9, 0.35, 0.6, 0.32, 0.77, 0.15, 0.82, 0.4, 0.68, 0.05]
    sentences = []
    docs_ids = ['C', 'A', 'B', 'A', 'B', 'D', 'A', 'C', 'D', 'B']
    ground_truth_id = 'D'
    metric = QAMetrics()
    print("k =", k, "Recall =", metric.calculate_recall(k, scores, sentences, docs_ids, ground_truth_id))
    print(metric.calculate_log_discount(scores,sentences, docs_ids))
