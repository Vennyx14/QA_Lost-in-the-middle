# src/retriever/search.py
import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

class FAISSRetriever:
    """
    Core Information Retrieval module using FAISS and BKAI Bi-encoder.
    Converts queries to vectors, applies L2 normalization, and retrieves top-K documents.
    """
    def __init__(self, 
                 model_name: str = "bkai-foundation-models/vietnamese-bi-encoder", 
                 index_path: str = "data/index/faiss_index.bin",
                 doc_map_path: str = "data/index/doc_map.pkl"):
        """
        Initialize the Retriever.
        
        Args:
            model_name (str): HuggingFace model path for Vietnamese Bi-Encoder.
            index_path (str): Path to the saved FAISS index (.bin).
            doc_map_path (str): Path to the pickle file mapping FAISS IDs to raw text.
        """
        print(f"[*] Loading Bi-Encoder model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        self.abs_index_path = os.path.abspath(index_path)
        self.abs_doc_map_path = os.path.abspath(doc_map_path)
        
        # Load FAISS Index
        if not os.path.exists(self.abs_index_path):
            print(f"[!] Warning: Index file not found at {self.abs_index_path}. Running in MOCK mode.")
            self.index = None
            self.doc_map = {}
        else:
            print("[*] Loading FAISS index...")
            self.index = faiss.read_index(self.abs_index_path)
            
            # Load Document Map (Dict mapping ID -> Text)
            with open(self.abs_doc_map_path, 'rb') as f:
                self.doc_map = pickle.load(f)
            print(f"[*] Successfully loaded {self.index.ntotal} vectors into memory.")

    def _normalize_l2(self, vector: np.ndarray) -> np.ndarray:
        """
        Applies L2 Normalization to a vector.
        Essential for converting FAISS Inner Product to Cosine Similarity.
        """
        norm = np.linalg.norm(vector, axis=1, keepdims=True)
        # Avoid division by zero
        norm[norm == 0] = 1e-10
        return vector / norm

    def search(self, query: str, top_k: int = 20) -> Tuple[List[str], List[float]]:
        """
        Perform semantic search for a given query.
        
        Args:
            query (str): The user's question.
            top_k (int): Number of documents to retrieve.
            
        Returns:
            Tuple containing:
            - List of raw document texts.
            - List of corresponding similarity scores.
        """
        # --- MOCK MODE ---
        if self.index is None:
            # Create dummy docs
            dummy_docs = [
                "Sinh viên UET ngành Trí tuệ nhân tạo cần hoàn thành 135 tín chỉ để tốt nghiệp. Trong đó có 10 tín chỉ thực tập doanh nghiệp. Các môn học toán rất nặng.",
                "Khoa học máy tính là một ngành hot tại Đại học Công nghệ. Điểm chuẩn năm ngoái lên tới 27 điểm. Sinh viên ra trường thường làm tại các tập đoàn lớn.",
                "Trường Đại học Công nghệ (VNU-UET) nằm ở 144 Xuân Thủy. Căn tin trường khá nhỏ. Thư viện có rất nhiều sách chuyên ngành."
            ]
            # Return these documents with the fake scores
            return dummy_docs[:top_k], [0.95, 0.85, 0.75][:top_k]
        # ------------------------------------------------------

        # 1. Encode query to vector
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # 2. L2 Normalization for Cosine Similarity 
        query_embedding = self._normalize_l2(query_embedding)
        
        # 3. FAISS Search
        # D: Distances/Scores (Cosine Similarities), I: Indices (IDs)
        scores, indices = self.index.search(query_embedding, top_k)
        
        # 4. Map IDs back to raw text
        retrieved_docs = []
        retrieved_scores = []
        
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1 and idx in self.doc_map:
                retrieved_docs.append(self.doc_map[idx])
                retrieved_scores.append(float(score))
                
        return retrieved_docs, retrieved_scores

# --- Unit Testing ---
if __name__ == "__main__":
    retriever = FAISSRetriever()
    q = "Sinh viên UET cần bao nhiêu tín chỉ để ra trường?"
    docs, scores = retriever.search(q, top_k=3)
    
    print(f"\nQuery: {q}")
    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"Rank {i+1} | Score: {score:.4f} | Text: {doc[:100]}...")