import json 


professions = ["doctor", "lawyer", "engineer", "writer", "janitor", "bartender"]

pairs = []
nicknames = {}

for p1 in professions:
    for p2 in professions: 
        if p1 == p2: 
            continue
        s1 = f"the {p1}"
        s2 = f"the {p2}" 
        pairs.append([s1, s2])
        nicknames[s1] = [s1.lower(), p1]
        nicknames[s2] = [s2.lower(), p2]

with open("../data/professions.json","w") as f1:
    json.dump(pairs, f1, indent=4) 
with open("../data/nicknames_professions.json","w") as f1:
    json.dump(nicknames, f1, indent=4) 
