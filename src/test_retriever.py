import os
import numpy as np

# Import 3 core modules
from src.data_prep.segmenter import VietnameseSegmenter
from src.retriever.embedding import VietnameseEmbedder
from src.retriever.faiss_index import FAISSRetriever

def main():
    """
    Main function to execute the end-to-end integration test of the Retriever pipeline.
    It simulates a mini-corpus, embeds it, builds the FAISS index, and performs a search.
    """
    print("      STARTING RETRIEVER PIPELINE INTEGRATION TEST")
    print("=" * 50)


    # STEP 1: Initialization

    print("\n[1] Initializing system modules...")
    segmenter = VietnameseSegmenter(save_dir="./vncorenlp_models")
    embedder = VietnameseEmbedder(model_name="bkai-foundation-models/vietnamese-bi-encoder")
    faiss_db = FAISSRetriever(vector_dim=768)


    # STEP 2: Define Dummy Corpus and Query

    print("\n[2] Loading dummy Vietnamese dataset...")
    raw_documents = [
        "Trường Đại học Công nghệ, Đại học Quốc gia Hà Nội được thành lập vào năm 2004.",
        "Trường Đại học Công nghệ là ngôi trường nằm trong top đầu của cả nước.",
        "Đại học Bách khoa thành lập năm 1956."
        "Các trường đều đang được cải tiến theo thời gian."
        "Học tăng cường (Reinforcement Learning) là một lĩnh vực của học máy, tập trung vào việc ra quyết định.",
        "Mô hình ngôn ngữ lớn (LLM) đòi hỏi hàng ngàn GPU để huấn luyện trong nhiều tháng.",
        "Để đánh giá một hệ thống truy xuất thông tin, các nhà nghiên cứu thường dùng độ đo MRR và Hit Rate."
    ]
    
    raw_query = "Đại học Công nghệ VNU thành lập năm nào?"
    print(f"Target Query: '{raw_query}'")


    # STEP 3: Text Segmentation (VnCoreNLP)

    print("\n[3] Segmenting text to handle Vietnamese compound words...")
    segmented_docs = [segmenter.word_segment(doc) for doc in raw_documents]
    segmented_query = segmenter.word_segment(raw_query)
    
    print(f"Segmented Query: '{segmented_query}'")


    # STEP 4: Dense Vector Embedding (BKAI)

    print("\n[4] Embedding the corpus into 768-dimensional dense vectors...")
    corpus_embeddings = embedder.encode_corpus(segmented_docs)
    print(f"Corpus Matrix Shape: {corpus_embeddings.shape}")


    # STEP 5: FAISS Indexing (Inner Product)
    print("\n[5] Building and saving the FAISS index...")
    
    faiss_db.build_index(corpus_embeddings)
    
    # Save the index to test the local storage capability
    index_path = os.path.join("data", "index", "dummy_corpus.bin")
    faiss_db.save_index(index_path)

    
    # STEP 6: Query Embedding and Similarity Search
    
    print("\n[6] Embedding the query and performing vector search...")
    
    query_vector = embedder.encode_query(segmented_query)
    
    # We want to retrieve the top k most relevant documents
    top_k = 4
    scores, indices = faiss_db.search(query_vector, top_k=top_k)

    # STEP 7: Display Final Results
    print("\n" + "=" * 50)
    print(f" SEARCH RESULTS FOR: '{raw_query}'")
    print("=" * 50)
    
    for rank, (score, doc_idx) in enumerate(zip(scores, indices), start=1):
        print(f"Rank {rank} | Vector Distance Score: {score:.4f}")
        print(f"Matched Document: {raw_documents[doc_idx]}")
        print("-" * 50)

if __name__ == "__main__":
    main()