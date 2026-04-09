import numpy as np
import re
from collections import Counter

class QAMetrics:
    """
    Lớp cung cấp các phương thức đo lường hiệu quả cho hệ thống RAG nén ngữ cảnh.
    """

    @staticmethod
    def normalize_text(text):
        """
        Chuẩn hóa văn bản: viết thường, xóa dấu câu và khoảng trắng thừa.
        Giúp việc so sánh giữa văn bản gốc và văn bản nén công bằng hơn.
        """
        text = text.lower()
        # Xóa các ký tự đặc biệt/dấu câu
        text = re.sub(r'[^\w\s_]', '', text) 
        # Xóa khoảng trắng thừa
        text = " ".join(text.split())
        return text

    @staticmethod
    def calculate_hit_rate(predicted_context, ground_truth_answer):
        """
        Hit Rate: Trả về 1 nếu đáp án nằm trong đoạn nén, ngược lại trả về 0.
        """
        pred = QAMetrics.normalize_text(predicted_context)
        truth = QAMetrics.normalize_text(ground_truth_answer)
        
        return 1 if truth in pred else 0

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
        MRR (Mean Reciprocal Rank): Dùng để đánh giá vị trí của văn bản đúng sau khi Bicoder lấy ra các tập văn bản liên quan nhất.
        relevance_list: List các giá trị 0 và 1. 
        Ví dụ: [0, 1, 0] nghĩa là đáp án nằm ở vị trí thứ 2.
        """
        for i, score in enumerate(relevance_list):
            if score > 0:
                return 1 / (i + 1)
        return 0

    @staticmethod
    def compression_stats(original_text_list, compressed_text):
        """
        Tính toán các chỉ số về nén.
        - compression_ratio: Tỷ lệ độ dài sau nén / trước nén (càng thấp nén càng nhiều).
        """
        original_full_text = " ".join(original_text_list)
        len_orig = len(original_full_text.split())
        len_comp = len(compressed_text.split())
        
        ratio = len_comp / len_orig if len_orig > 0 else 0
        return {
            "original_tokens": len_orig,
            "compressed_tokens": len_comp,
            "compression_ratio": round(ratio, 4)
        }

# Ví dụ sử dụng
if __name__ == "__main__":
    query_answer = "Thủ đô của Việt Nam là Hà Nội"
    # Giả sử sau khi nén 20 đoạn văn thu được kết quả này
    compressed_result = "Hà Nội là thủ đô của Việt Nam, một thành phố có lịch sử lâu đời..."
    
    metrics = QAMetrics()
    
    hit = metrics.calculate_hit_rate(compressed_result, query_answer)
    f1 = metrics.calculate_f1(compressed_result, query_answer)
    
    print(f"Hit Rate: {hit}") # Output: 1
    print(f"F1-Score: {f1:.4f}")