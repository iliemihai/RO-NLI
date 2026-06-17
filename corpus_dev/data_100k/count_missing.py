import json
from tqdm import tqdm

d = json.load(open("ro_train_sentence2id.json", "r"))
indices = [x[1]  for x in d]

new_lines = []
for i in tqdm(range(200192, 519213)):
    if i not in indices:
        new_lines.append(i)

print(new_lines)
