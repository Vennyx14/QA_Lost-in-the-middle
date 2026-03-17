import faiss
import numpy as np
import os

class FAISSRetriever:
    """
    A class to manage the FAISS vector database using the Inner Product (IP) algorithm.
    """
    def __init__(self, vector_dim: int = 768):
        self.vector_dim = vector_dim
        # IndexFlatIP calculates the Inner Product. 
        # When vectors are L2-normalized, Inner Product mathematically equals Cosine Similarity.
        self.index = faiss.IndexFlatIP(self.vector_dim)
    
    def build_index(self,corpus_embedding = np.ndarray):
        """
        Normalize and add document embeddings to the FAISS index in RAM.
        
        Args:
            corpus_embeddings (np.ndarray): The 2D numpy array of document vectors.
        """
        if corpus_embedding.dtype != np.float32:
            corpus_embedding = corpus_embedding.astype(np.float32)
        
        #L2 normalization
        faiss.normalize_L2(corpus_embedding)
        self.index.add(corpus_embedding)
        print(f"[*] FAISS Index built successfully. Total vectors indexed: {self.index.ntotal}")

    def save_index(self, save_path: str):
        """
        Save the FAISS index from RAM to the local hard drive.
        
        Args:
            save_path (str): The file path to save the index (e.g., 'data/index/corpus.bin').
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        faiss.write_index(self.index, save_path)
        print(f"[*] FAISS Index successfully saved to {save_path}")

    def load_index(self, load_path: str):
        """
        Load the FAISS index from the local hard drive back into RAM.
        
        Args:
            load_path (str): The file path of the saved index.
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"FAISS index file not found at {load_path}")
            
        self.index = faiss.read_index(load_path)
        print(f"[*] FAISS Index loaded. Total vectors ready for search: {self.index.ntotal}")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> tuple:
        """
        Search for the top-K most similar documents to the given query vector.
        
        Args:
            query_vector (np.ndarray): The 1D query vector.
            top_k (int): The number of top documents to retrieve.
            
        Returns:
            tuple: Two lists containing (similarity_scores, document_indices).
        """
        # Reshape the 1D vector to 2D matrix (1 row, 768 columns) as strictly required by FAISS
        query_matrix = query_vector.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_matrix)
        
        # Perform the search
        scores, indices = self.index.search(query_matrix, top_k)
        
        # Flatten the results back to standard Python 1D lists
        return scores[0].tolist(), indices[0].tolist()