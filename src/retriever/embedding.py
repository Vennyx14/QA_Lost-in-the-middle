import torch 
import numpy as np
from sentence_transformers import SentenceTransformer

class VietnameseEmbedder:
    """
    A class to handle embedding using the pre-trained BKAI Vietnamese Bi-Encoder.
    """
    def __init__(self, model_name: str = "bkai-foundation-models/vietnamese-bi-encoder"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name).to(self.device)
        print(f"[*]Embedding model '{model_name} load successfully on '{self.device}")
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single search query into a dense vector.
        
        Args:
            query (str): The segmented query string.
            
        Returns:
            np.ndarray: A 1D numpy array representing the query vector (float32).
        """
        # convert_to_numpy=True bypasses the need to manually detach PyTorch tensors.
        vector = self.model.encode(query, convert_to_numpy= True).astype(np.float32)
        return vector
    
    def encode_corpus(self, documents: list) -> np.ndarray:
        """
        Encode a list of documents into a 2D matrix of vectors.
        
        Args:
            documents (list): A list of segmented document strings.
            
        Returns:
            np.ndarray: A 2D numpy array representing the document vectors (float32).
        """
        print(f"[*]Encoding {len(documents)}")
        vectors = self.model.encode(
            documents,
            convert_to_numpy= True,
            show_progress_bar= True,
            batch_size=32
        ).astype(np.float32)
        return vectors
