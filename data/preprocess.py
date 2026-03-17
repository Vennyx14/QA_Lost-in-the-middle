import json

#mmarco-EnVi
mmarco_in = "data/raw/mmarco-EnVi-subdataset/train.jsonl"

#tydiqa-goldp
corpuses_in = "data/raw/tydiqa-goldp-vietnamese/corpuses.jsonl"
queries_in = "data/raw/tydiqa-goldp-vietnamese/queries.jsonl"


#UIT-ViQuAD2.0
uit_in = ["data/raw/UIT-ViQuAD2.0/train.jsonl", "data/raw/UIT-ViQuAD2.0/validation.jsonl"]

output = "data/processed/processed_data.jsonl"

if __name__ == "__main__":
    with open(output, "w", encoding="utf-8") as f_out:
        with open (mmarco_in, "r", encoding="utf-8") as f_in:
            for line in f_in:
                item = json.loads(line)
                query = item["query"]
                answer = item["positive"]
                f_out.write(json.dumps({"query": query, "answer": answer}, ensure_ascii=False) + "\n")

        corpus_dict = {}
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
                if answer:
                    f_out.write(json.dumps({"query":query, "answer": answer}, ensure_ascii=False)+ "\n")
        
        for input_file in uit_in:
            with open(input_file, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    item = json.loads(line)
                    query = item["question"]
                    answers = item.get("answers", {})
                    plausible_ans = item.get("plausible_answers", {})

                    if answers.get("text"):
                        answer = answers["text"][0]
                    elif plausible_ans.get("text"):
                        answer = plausible_ans["text"][0]
                    else:
                        continue

                    f_out.write(json.dumps({"query":query, "answer": answer}, ensure_ascii=False) + "\n")
