# src/data_prep/splitter.py
import py_vncorenlp
import os
import re
from typing import List, Optional

class SentenceSplitter:
    """
    A module dedicated to cutting long texts into individual sentences.
    Supports 2 modes:
    1. VnCoreNLP: Slower but absolutely accurate, comes with Word Segment support for PhoBERT.
    2. Regex: High speed, used for testing workflows or processing raw text.
    """
    def __init__(self, 
                 use_vncorenlp: bool = True, 
                 save_dir: str = './vncorenlp_models',
                 vncorenlp_instance: Optional[py_vncorenlp.VnCoreNLP] = None):
        """
        Initialize the sentence splitter.

        Args:
        use_vncorenlp (bool): Enable/disable using VnCoreNLP.
        save_dir (str): Path to the folder containing the model.
        vncorenlp_instance: Receive a previously initialized instance to avoid RAM overflow (Memory Optimization).
        """
        self.use_vncorenlp = use_vncorenlp
        self.segmenter = None
        
        if self.use_vncorenlp:
            # Optimization: If there is already an instance passed from the main file, use it to save RAM 
            if vncorenlp_instance is not None:
                self.segmenter = vncorenlp_instance
                print("[*] SentenceSplitter is reusing the provided VnCoreNLP instance.")
            else:
                self.abs_save_dir = os.path.abspath(save_dir)
                jar_path = os.path.join(self.abs_save_dir, "VnCoreNLP-1.2.jar")
                
                if not os.path.exists(jar_path):
                    raise FileNotFoundError(
                        f"Missing VnCoreNLP-1.2.jar in {self.abs_save_dir}. "
                        "Ensure you placed the .jar file and 'models' folder here."
                    )
                
                print(f"[*] SentenceSplitter initializing JVM with classpath: {jar_path}")
                self.segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=self.abs_save_dir)
                print("[*] VnCoreNLP loaded successfully for Sentence Splitting.")

    def split(self, text: str, min_words: int = 3) -> List[str]:
        """
        Split text into a list of simple sentences.

        Args:
        text (str): The original text to split.
        min_words (int): Filter out sentences that are too short (noise).

        Returns:
        List[str]: List of sentences.
        """
        if not text or not isinstance(text, str):
            return []
            
        if self.use_vncorenlp and self.segmenter is not None:
            try:
                sentences = self.segmenter.word_segment(text)
                return [s for s in sentences if len(s.split()) >= min_words]
            except Exception as e:
                print(f"[!] Error when running VnCoreNLP split: {e}.Switching to Regex fallback...")
                return self._regex_split(text, min_words)
        else:
            return self._regex_split(text, min_words)

    def _regex_split(self, text: str, min_words: int) -> List[str]:
        """
        The fallback function uses Regex to quickly split sentences.
        Uses lookbehind to keep the punctuation.
        """

        raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        cleaned_sentences = []
        for s in raw_sentences:
            s = s.strip()

            if len(s.split()) >= min_words:
                cleaned_sentences.append(s)
                
        return cleaned_sentences

# --- Unit Testing ---\r
if __name__ == "__main__":
    # Test 1: Run Regex
    print("--- TEST REGEX MODE ---")
    fast_splitter = SentenceSplitter(use_vncorenlp=False)
    sample_text = "Trí tuệ nhân tạo (AI) đang phát triển rất nhanh! Đặc biệt là các mô hình LLM. Tuy nhiên, hiện tượng Lost in the Middle vẫn là một thách thức lớn. K.H.T.N là viết tắt của Khoa học tự nhiên."
    
    fast_results = fast_splitter.split(sample_text)
    for i, s in enumerate(fast_results):
        print(f"Câu {i+1}: {s}")
        
    print("\n--- TEST VNCORENLP MODE ---")
    # Test 2: VnCoreNLP
    try:

        accurate_splitter = SentenceSplitter(use_vncorenlp=True)
        accurate_results = accurate_splitter.split(sample_text)
        for i, s in enumerate(accurate_results):
            print(f"Câu {i+1}: {s}")
    except Exception as e:
        print(f"Skip VnCoreNLP test due to configuration error: {e}")