import json

#df1 = json.load(open("train_sentence2id.json", "r"))
#dic1 = {x[1]: x[0] for x in df1}

df2 = json.load(open("ro_dev_mismatched_sentence2id.json", "r"))
dic2 = {x[1]: x[0] for x in df2}

map = json.load(open("dev_mismatched_mapping.json", "r"))
print(len(map))
#g = open("mnli.tsv", "w")
#g.write("guid\tsentence1\tsentence_2\tcompare\n")


#for key, values in zip(map.keys(), map.values()):
#    if key in "6802e_53704":
#        break;
#    g.write(key+"\t"+dic1[values[0]]+"\t"+dic1[values[1]]+"\t"+values[2]+"\n")


g = open("dev_mismatched.tsv", "w")
g.write("guid\tsentence1\tsentence_2\tcompare\n")

for key, values in zip(map.keys(), map.values()):
    #if values[0] < 200192 and values[1] < 200192:
        #if key in "6802e_53704":
        #    break;
    try:
        g.write(key+"\t"+dic2[values[0]]+"\t"+dic2[values[1]]+"\t"+values[2]+"\n")
    except Exception as e:
        print(e)
