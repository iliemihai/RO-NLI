import os, sys, json
import argparse

parser = argparse.ArgumentParser(description='Create Doccano import file script')
parser.add_argument('-l', type=int, nargs=2, help='start stop line #')
parser.add_argument('-c', type=int, default=1, help='# of chunks to split into')
args = parser.parse_args()

with open("sentence2id.json", "r", encoding="utf8") as f:
    sentence2id = json.load(f)
    id2sentence = []
    for sentence in sentence2id:
        id2sentence.append(sentence[0])

with open("mapping.json", "r", encoding="utf8") as f:
    mapping = json.load(f)

print("Unique sents: {}, total sentences: {}".format(len(sentence2id), len(mapping)))

count = int( (args.l[1] - args.l[0])/args.c )
print("Extracting lines {} - {} (last line excluded) and splitting into {} chunk(s), each with {} lines...".format(args.l[0], args.l[1], args.c, count))

if (args.l[1] - args.l[0]) % args.c != 0:
    print("Ensure that line count divides neatly by # of chunks!")
    sys.exit(0)



for i in range(args.c):
    filename = "doccano_import_{}.jsonl".format(i)
    with open(filename,"w", encoding="utf8") as f:
        cnt=0
        start = args.l[0] + i*count
        end = args.l[0] + (i+1)*count
        print("Writing lines {} to {} in file {}".format(start, end, filename))
        for index in range(start, end):
            obj = {
                "text": id2sentence[index],
                "labels": [],
                "meta": {"sent_id":index}
            }
            f.write(str(json.dumps(obj))+"\n")