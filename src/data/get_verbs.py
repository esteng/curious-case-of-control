import json
import csv

lookup = dict()
with open('../data/openwebtext-clean2.num') as csvf:
    f = csv.reader(csvf, delimiter="\t")

    for line in f:
        word = line[0]
        freq = line[2]
        lookup[word] = int(freq)

verbs_by_freq = []
with open("../data/verbs.json") as f1:
    verbs = json.load(f1)
    for __, verb in verbs:
        verbs_by_freq.append((verb, lookup[verb]))

sorted_verbs = sorted(verbs_by_freq, key=lambda x:x[1])

top_20_verbs = sorted_verbs[-20:]
print(top_20_verbs)

# verb list: exclude things that are also nouns and abstract things
# ('call', 2041225), ('open', 2210459), ('live', 2247889), ('run', 2280036), ('feel', 2520384), ('read', 3327909), ('come', 3806126), ('go', 5617348)
# keep "go", "come", "read", "run", "call"