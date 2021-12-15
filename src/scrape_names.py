import re 
import itertools
from collections import defaultdict

import requests
from bs4 import BeautifulSoup


def parse_row(row): 
    entries = row.find_all("td")
    entries = [x.string for x in entries]
    man_name, man_count, female_name, female_count  = entries[1:]
    man_count = re.sub(",","", man_count)
    female_count = re.sub(",","", female_count)
    return man_name, man_count, female_name, female_count 

def get_data(decade):
    url = f"https://www.ssa.gov/oact/babynames/decades/names{decade}s.html"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    rows = soup.find_all("tr")

    rows = soup.find_all("tr")
    counts = {"male": defaultdict(int), "female": defaultdict(int)}
    for row in rows[2:]:
        try:
            man_name, man_count, female_name, female_count = parse_row(row)
            counts["male"][man_name] += int(man_count)
            counts["female"][female_name] += int(female_count)
        except ValueError:
            break
    return counts

def get_top(decades = ["1970", "1980", "1990", "2000", "2010"], top_k=10):
    all_data = []
    for decade in decades:
        all_data.append(get_data(decade))
    
    # merge 
    big_counts = {"male": defaultdict(int), "female": defaultdict(int)}
    for small_counts in all_data:
        for key in ["male", "female"]:
            for name in small_counts[key].keys():
                big_counts[key][name] += small_counts[key][name]

    # take top male and female
    male_names_and_counts = big_counts['male']
    female_names_and_counts = big_counts['female']
    sorted_male = sorted(male_names_and_counts.items(), key=lambda x:x[1])
    sorted_female = sorted(female_names_and_counts.items(), key=lambda x:x[1])

    top_male = sorted_male[-top_k:]
    top_female = sorted_female[-top_k:]

    both_names = set(male_names_and_counts.keys()) & set(female_names_and_counts.keys())
    both_counts = {k: male_names_and_counts[k] + female_names_and_counts[k] for k in both_names}
    sorted_neutral = sorted(both_counts.items(), key=lambda x:x[1])
    top_neutral = sorted_neutral[-top_k:]

    return top_male, top_female, top_neutral 





if __name__ == "__main__":
    top_male, top_female, top_neutral = get_top()
    print(f"top male: {top_male}")
    print(f"top female: {top_female}")
    print(f"top neutral : {top_neutral}")


    top_2_male = [x[0] for x in top_male[0:2]]
    top_2_female = [x[0] for x in top_female[0:2]]
    top_2_neutral = [x[0] for x in top_neutral[0:2]]

    top_names = top_2_male + top_2_female + top_2_neutral

    combos = list(set(itertools.product(top_names, repeat=2)))
    # combos = list(itertools.product(top_2_male, top_2_female)) + list(itertools.product(top_2_male, top_2_neutral)) + list(itertools.product(top_2_male, top_2_male)) + \
            #  list(itertools.product(top_2_female, top_2_female)) + list(itertools.product(top_2_neutral, top_2_neutral))
    # combos = list(set(combos))
    combos = [x for x in combos if x[0] != x[1]]
    combos = sorted(combos)
    
    
    print(combos)

