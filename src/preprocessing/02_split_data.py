import json
import random
import os

"""
    Gộp mmarco, uit, chia thành 2 tập: train - valid - test (85 - 15)
"""

def merge_split_data():
    input_file = ["dataset/preprocessed/pr_mmarco.jsonl", "dataset/preprocessed/pr_uit.jsonl"]
    output_train = "dataset/preprocessed/train.jsonl"
    output_valid = "dataset/preprocessed/validation.jsonl"

    combined_data = []
    for file in input_file:
        if os.path.exists(file):
            with open(file, "r", encoding = "utf-8") as f:
                for line in f:
                    combined_data.append(json.loads(line))
    total_record = len(combined_data)
    print("total_record:", total_record)

    if total_record == 0:
        return
    random.seed(42)
    random.shuffle(combined_data)
    train_end = int(total_record * 0.85)
    # valid_end = int(total_record * 0.1) + train_end
    print("train data:", int(total_record*0.85))
    print("valid data:", int(total_record*0.15))


    train_data = combined_data[:train_end]
    valid_data = combined_data[train_end:]

    def save_to_jsonl(data, output):
        with open(output, "w", encoding = "utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    save_to_jsonl(train_data, output_train)
    save_to_jsonl(valid_data, output_valid)

if __name__ == "__main__":
    merge_split_data()