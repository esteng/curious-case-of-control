import pathlib 
import re 
import pdb
import json
import argparse 
import numpy as np
from tqdm import tqdm 
from datasets import load_dataset 
np.random.seed(12) 

test_sents = ["Mary promised Tom to come home",
              "Mary threatened Tome to come home",
              "Mary felt threatened by tom",
              "I am a person who told someone off",
              "I told Roger to get out",
              "I was promised a French baguette",
              "I called upon the barber to bear arms",
              "The lawyer persuaded the client to plead guilty",
              "The client promised the lawyer to plead guilty",
              "Tom convinced Mary that the moon was made of cheese"]

def get_sample(args):
    if pathlib.Path(args.sample_path).exists() and not args.overwrite:
        print(f"loading from {args.sample_path}") 
        with open(args.sample_path) as f1:
            sample = json.load(f1) 
            assert(len(sample) == args.sample_size)
        return sample 

    print(f"loading from dataset...") 
    dataset = load_dataset("allenai/c4", cache_dir="/brtx/604-nvme2/estengel/.cache/", split="train", streaming=True)
    sample = dataset.take(args.sample_size) 
    sample = [{"text": ex['text']} for ex in sample]

    with open(args.sample_path, "w") as f1:
        json.dump(sample, f1) 

    return sample 

def get_fake_sample(): 
    return [{"text": s} for s in test_sents]

def get_regex():
    oc_verbs = ["told", "ordered", "called upon", "reminded", "urged", "asked", "persuaded", "convinced", "forced", "pushed"]
    sc_verbs = ["promised", "threatened"]
    oc_verbs = [f"({v})" for v in oc_verbs]
    sc_verbs = [f"({v})" for v in sc_verbs]
    oc_verb_disj = f"({'|'.join(oc_verbs)})"
    sc_verb_disj = f"({'|'.join(sc_verbs)})"
    oc_regex = re.compile(f"\w+ {oc_verb_disj} ((the )|(a ))?\w+ to \w+")
    sc_regex = re.compile(f"\w+ {sc_verb_disj} ((the )|(a ))?\w+ to \w+") 
    return oc_regex, sc_regex

def main(args): 
    if args.debug:
        sample = get_fake_sample() 
    else:
        sample = get_sample(args) 
    oc_regex, sc_regex = get_regex()  

    sc_count, oc_count = 0,0 
    for example in tqdm(sample):
        sent = example['text'] 
        oc_match = oc_regex.search(sent)
        sc_match = sc_regex.search(sent) 
            
        if oc_match is not None:
            oc_count += 1
        if sc_match is not None:
            sc_count += 1

    print(f"In sample of {args.sample_size}, there are {oc_count} instances of object control and {sc_count} instances of subject control") 

def test(): 
    oc_regex, sc_regex = get_regex()

    for sent in test_sents:
        oc_match = oc_regex.search(sent) 
        sc_match = sc_regex.search(sent) 
        if oc_match is None:
            oc_match_str = "None"
        else:
            oc_match_str = oc_match.group(0) 
        if sc_match is None:
            sc_match_str = "None" 
        else:
            sc_match_str = sc_match.group(0) 

        
        print(f"Sentence: {sent}\nObject control: {oc_match_str}\nSubject control: {sc_match_str}\n") 


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=1000000)
    parser.add_argument("--sample-path", type=str, default="../data/sample.json")
    parser.add_argument("--debug", action="store_true") 
    parser.add_argument("--overwrite", action="store_true") 
    args = parser.parse_args() 

    #test() 
    
    main(args) 






