# src/data_prep/segmenter.py
import py_vncorenlp
import os

class VietnameseSegmenter:
    """
    A wrapper class for Vietnamese word segmentation using VnCoreNLP.
    """
    def __init__(self, save_dir: str = './vncorenlp_models'):
        """
        Initialize the VnCoreNLP segmenter for word segmentation only.
        
        Args:
            save_dir (str): The relative or absolute path containing the .jar and models.
        """
        # os.path.abspath() converts './vncorenlp_models' to absolute path
        self.abs_save_dir = os.path.abspath(save_dir)
        
        # Validate that the manual download was completed
        jar_path = os.path.join(self.abs_save_dir, "VnCoreNLP-1.2.jar")
        if not os.path.exists(jar_path):
            raise FileNotFoundError(
                f"Missing VnCoreNLP-1.2.jar in {self.abs_save_dir}. "
                "Ensure you placed the .jar file and 'models' folder here."
            )
        
        print(f"[*] Initializing JVM with classpath: {jar_path}")
        
        # Load the model using the absolute path
        self.rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=self.abs_save_dir)
        print("[*] VnCoreNLP word segmenter loaded successfully.")

    def word_segment(self, text: str) -> str:
        """
        Perform word segmentation on the input text.
        
        Args:
            text (str): The raw Vietnamese text.
            
        Returns:
            str: The segmented text with compound words joined by underscores.
        """
        segmented_sentences = self.rdrsegmenter.word_segment(text)
        return " ".join(segmented_sentences)

# --- Unit Testing ---
if __name__ == "__main__":
    segmenter = VietnameseSegmenter()
    
    sample_text = "Ông Nguyễn Khắc Chúc đang làm việc tại Đại học Quốc gia Hà Nội."
    print(f"[Input] Original text: {sample_text}")
    print(f"[Output] Segmented text: {segmenter.word_segment(sample_text)}")