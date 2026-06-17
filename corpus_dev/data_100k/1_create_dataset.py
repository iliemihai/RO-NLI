import json
from tqdm import tqdm

#train_mapping.json
#200k.json

dict_data = json.load(open("dev_mismatched_mapping.json", "r"))

list_data = json.load(open("ro_dev_mismatched_sentence2id.json", "r"))

list_data_dict = { x[1]:x[0] for x in list_data}

#for i in range(500000):
#    if i not in list_data_dict.keys():
#        print(i)
json_dataset = []

# Placeholder logic for matching and transformation
# This needs to be adapted based on how the dictionary keys relate to the list indices or contents
for dict_key, dict_values in tqdm(dict_data.items()):
    # Extract data from dictionary
    idx1, idx2, label, _ = dict_values
    sentence_premise = list_data_dict[idx1]
    sentence_hypothesis = list_data_dict[idx2]
    
    # Find matching entry in the list based on some condition
    #for entry in list_data:
    #    sentence, list_idx = entry
        
    #    # Match based on index or a transformation of the dict_key
    #    # Placeholder condition: match if list index matches the first value of the dictionary entry
    #    if list_idx == idx:
    #        # Construct the JSON object for this match
    json_entry = {
                "promptID": dict_key,  # Assuming this is the promptID
                #"pairID": dict_key + "n",  # Assuming we simply append 'n' for the pairID
                "premise": sentence_premise,  # Assuming the sentence from the list is the premise
                "hypothesis": sentence_hypothesis,  # The example does not clarify how to obtain the hypothesis
                "label": label
    }
    json_dataset.append(json_entry)
    #        #break  # Assuming each list entry matches at most one dictionary entry

#print(json_dataset)  # Display the constructed dataset

file_path = "ro_mismatched_mnli.json"

# Writing JSON data to file
with open(file_path, "w", encoding='utf-8') as file:
    json.dump(json_dataset, file, ensure_ascii=False, indent=4)
