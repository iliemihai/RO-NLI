"""
	Please ensure snli_1.0 folder is extracted here and contains the train dev and test tab-separated files
"""
import os, sys, json


def process_lines(lines, sentence2id, mapping, type):
    for ind, line in enumerate(lines[1:]): # skip header
        parts = line.strip().split("\t")
        s1 = parts[5]
        s2 = parts[6]
        gold_label = parts[0]
        pair_id = "{0}_{1}".format(parts[8], ind)

        if s1 not in sentence2id:
            sentence2id[s1] = len(sentence2id)
        if s2 not in sentence2id:
            sentence2id[s2] = len(sentence2id)
        assert pair_id not in mapping, "sanity check"
        mapping[pair_id] = [sentence2id[s1], sentence2id[s2], gold_label, type]

    return sentence2id, mapping


def count(folder="train", source_path="snli_1.0/snli_1.0"):
    sentence2id = {}
    mapping = {}
    
    with open(f"{source_path}_{folder}.txt","r", encoding="utf8") as f:
        sentence2id, mapping = process_lines(f.readlines(), sentence2id, mapping, folder)
        print("Unique sents: {}, mapping len: {}".format(len(sentence2id), len(mapping)))
    return sentence2id, mapping


def save_to_file(sentence2id, mapping, folder="train", save_path="snli_data"):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    with open(os.path.join(save_path, f"{folder}_sentence2id.json"), "w", encoding="utf8") as f:
        # convert to list to ensure order
        sentence2id_list = []
        for key, value in sentence2id.items():
            temp = [key, value]
            sentence2id_list.append(temp)
        json.dump(sentence2id_list, f, indent=2)

    with open(os.path.join(save_path, f"{folder}_mapping.json"), "w", encoding="utf8") as f:
        json.dump(mapping, f, indent=2)


print("Processing train SNLI dataset...")
snli_sentence2id, snli_mapping = count(folder="train", source_path="snli_1.0/snli_1.0")
save_to_file(snli_sentence2id, snli_mapping, folder="train", save_path="snli_data")

print("Processing dev SNLI dataset...")
snli_sentence2id, snli_mapping = count(folder="dev", source_path="snli_1.0/snli_1.0")
save_to_file(snli_sentence2id, snli_mapping, folder="dev", save_path="snli_data")

print("Processing test SNLI dataset...")
snli_sentence2id, snli_mapping = count(folder="test", source_path="snli_1.0/snli_1.0")
save_to_file(snli_sentence2id, snli_mapping, folder="test", save_path="snli_data")



print("Processing train MNLI dataset...")
mnli_sentence2id, mnli_mapping = count(folder="train", source_path="multinli_1.0/multinli_1.0")
save_to_file(mnli_sentence2id, mnli_mapping, folder="train", save_path="mnli_data")

print("Processing dev MNLI dataset...")
mnli_sentence2id, mnli_mapping = count(folder="dev_mismatched", source_path="multinli_1.0/multinli_1.0")
save_to_file(mnli_sentence2id, mnli_mapping, folder="dev_mismatched", save_path="mnli_data")

print("Processing test MNLI dataset...")
mnli_sentence2id, mnli_mapping = count(folder="dev_matched", source_path="multinli_1.0/multinli_1.0")
save_to_file(mnli_sentence2id, mnli_mapping, folder="dev_matched", save_path="mnli_data")
