import json
import random
from tqdm import tqdm


data1 = json.load(open("data.json", "r"))

data2 = json.load(open("train_sentence2id.json", "r"))

g = open("compares.txt", "w")
sentences = []
for i in tqdm(range(len(data1))):
    for j in range(len(data2)):
        if data1[i][1] == data2[j][1]:
            sentences.append(data1[i][0] + "\nVS\n " + data2[j][0]+"\n")
            break;

selected_sentences = random.sample(sentences, 100)
for i in tqdm(range(len(selected_sentences))):
    g.write(str(i)+"\n"+selected_sentences[i]+"\n")
