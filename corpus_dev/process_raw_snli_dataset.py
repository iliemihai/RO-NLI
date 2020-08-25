"""
	Please ensure snli_1.0 folder is extracted here and contains the train dev and test tab-separated files
"""
import os, sys, json


def process_lines(lines, sentence2id, mapping, type):
    for line in lines[1:]: # skip header
        parts = line.strip().split("\t")
        s1 = parts[5]
        s2 = parts[6]
        gold_label = parts[0]
        pair_id = parts[8]

        if s1 not in sentence2id:
            sentence2id[s1] = len(sentence2id)
        if s2 not in sentence2id:
            sentence2id[s2] = len(sentence2id)
        assert pair_id not in mapping, "sanity check"
        mapping[pair_id] = [sentence2id[s1], sentence2id[s2], gold_label, type]

    return sentence2id, mapping


sentence2id = {}
mapping = {}

with open("snli_1.0/snli_1.0_train.txt","r", encoding="utf8") as f:
    sentence2id, mapping = process_lines(f.readlines(), sentence2id, mapping, "train")
    print("Unique sents: {}, mapping len: {}".format(len(sentence2id), len(mapping)))

with open("snli_1.0/snli_1.0_dev.txt","r", encoding="utf8") as f:
    sentence2id, mapping = process_lines(f.readlines(), sentence2id, mapping, "dev")
    print("Unique sents: {}, mapping len: {}".format(len(sentence2id), len(mapping)))

with open("snli_1.0/snli_1.0_test.txt","r", encoding="utf8") as f:
    sentence2id, mapping = process_lines(f.readlines(), sentence2id, mapping, "test")
    print("Unique sents: {}, mapping len: {}".format(len(sentence2id), len(mapping)))

with open("sentence2id.json", "w", encoding="utf8") as f:
    # convert to list to ensure order
    sentence2id_list = []
    for key, value in sentence2id.items():
        temp = [key, value]
        sentence2id_list.append(temp)
    json.dump(sentence2id_list, f, indent=2)

with open("mapping.json", "w", encoding="utf8") as f:
    json.dump(mapping, f, indent=2)