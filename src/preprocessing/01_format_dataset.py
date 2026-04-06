import json
from langdetect import detect

"""
    Đưa các dataset raw về định dạng 1 question 1 answer, gộp vào thành 1 file duy nhất
"""

#mmarco-EnVi
mmarco_in = "dataset/raw/mmarco-EnVi-subdataset/train.jsonl"

#tydiqa-goldp
corpuses_in = "dataset/raw/tydiqa-goldp-vietnamese/corpuses.jsonl"
queries_in = "dataset/raw/tydiqa-goldp-vietnamese/queries.jsonl"


#UIT-ViQuAD2.0
uit_in = ["dataset/raw/UIT-ViQuAD2.0/train.jsonl", "dataset/raw/UIT-ViQuAD2.0/validation.jsonl"]

mmarco_out = "dataset/preprocessed/pr_mmarco.jsonl"
goldp_out = "dataset/preprocessed/pr_goldp_test.jsonl"
uit_out = "dataset/preprocessed/pr_uit.jsonl"

# output = "dataset/preprocessed/processed_data.jsonl"

if __name__ == "__main__":

    with open (mmarco_in, "r", encoding="utf-8") as f_in, open(mmarco_out, "w", encoding="utf-8") as f_out:
        for line in f_in:
            item = json.loads(line)
            query = item["query"]
            answer = item["positive"]
            if len(answer) < 200:
                continue
            if detect(answer) == 'en':
                continue
            f_out.write(json.dumps({"query": query, "answer": answer}, ensure_ascii=False) + "\n")

    corpus_dict = {}
    with open(goldp_out, "w", encoding="utf-8") as f_out:
        with open(corpuses_in, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                corpus_dict[item["passage_id"]] = item["passage"]

        with open(queries_in, "r", encoding = "utf-8") as f_in:
            for line in f_in:
                item = json.loads(line)
                query = item["question_text"]
                id = item["passage_id"]
                answer = corpus_dict.get(id)
                if len(answer) < 200:
                    continue
                if answer:
                    f_out.write(json.dumps({"query":query, "answer": answer}, ensure_ascii=False)+ "\n")
    
    with open(uit_out, "w", encoding="utf-8") as f_out:
        for input_file in uit_in:
            with open(input_file, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    item = json.loads(line)
                    query = item["question"]
                    answer = item['context']
                    if len(answer) < 200:
                        continue
                    f_out.write(json.dumps({"query":query, "answer": answer}, ensure_ascii=False) + "\n")
