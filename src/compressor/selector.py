import torch
import numpy as np
from transformers import AutoTokenizer

class ContextSelector:
    def __init__(self, model_path="vinai/phobert-base", token_limit=256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.token_limit = token_limit

    def get_token_count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def strategy_single_sentences(self, sentences, scores):
        """
        Strategy 1: Pick individual highest-scoring sentences.
        """
        # Sort by score descending
        ranked_indices = np.argsort(scores)[::-1]
        selected_sentences = []
        current_tokens = 0
        
        for idx in ranked_indices:
            sent = sentences[idx]
            count = self.get_token_count(sent)
            if current_tokens + count <= self.token_limit:
                selected_sentences.append((idx, sent))
                current_tokens += count
        
        # Sort back by original index to maintain document flow
        selected_sentences.sort(key=lambda x: x[0])
        return " ".join([s[1] for s in selected_sentences])

    def strategy_sentence_windows(self, sentences, scores, window_size=2):
        """
        Strategy 2: Group N sentences together as one unit.
        """
        windows = []
        window_scores = []
        
        for i in range(len(sentences) - window_size + 1):
            window_text = " ".join(sentences[i : i + window_size])
            # Average score of sentences in the window
            avg_score = np.mean(scores[i : i + window_size])
            windows.append(window_text)
            window_scores.append(avg_score)
            
        # Re-use logic from Strategy 1 to pick best windows
        # Note: You need to handle overlaps for a more advanced version
        return self.strategy_single_sentences(windows, window_scores)

    def strategy_mean_pooling(self, sentence_embeddings):
        """
        Strategy 3: Average embeddings to represent document (Baseline).
        This is useful for ranking docs before compression.
        """
        return torch.mean(torch.stack(sentence_embeddings), dim=0)

    def apply_softmax_scoring(self, raw_scores):
        """
        Strategy 4: Normalize scores to a probability distribution.
        """
        return torch.nn.functional.softmax(torch.tensor(raw_scores), dim=0).numpy()

    def select(self, method, **kwargs):
        if method == "single":
            return self.strategy_single_sentences(kwargs['sentences'], kwargs['scores'])
        elif method == "window":
            return self.strategy_sentence_windows(kwargs['sentences'], kwargs['scores'], kwargs.get('window_size', 2))
        # Add more methods as needed