import json
import random
import os

def merge_split_data():
    input_file = "data/processed/processed_data.jsonl"
    output_train = "data/processed/train.jsonl"
    output_valid = "data/processed/validation.jsonl"
    output_test = "data/processed/test.jsonl"

    combined_data = []
    if os.path.exists(input_file):
        with open(input_file, "r", encoding = "utf-8") as f:
            for line in f:
                combined_data.append(json.loads(line))
    total_record = len(combined_data)
    # print("total_record:", total_record)

    if total_record == 0:
        return
    random.seed(42)
    random.shuffle(combined_data)
    train_end = int(total_record * 0.8)
    valid_end = int(total_record * 0.1) + train_end
    # print("train data:", int(total_record*0.8))
    # print("valid/test data:", int(total_record*0.1))


    train_data = combined_data[:train_end]
    valid_data = combined_data[train_end:valid_end]
    test_data = combined_data[valid_end:]

    def save_to_jsonl(data, output):
        with open(output, "w", encoding = "utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    save_to_jsonl(train_data, output_train)
    save_to_jsonl(valid_data, output_valid)
    save_to_jsonl(test_data, output_test)

if __name__ == "__main__":
    merge_split_data()