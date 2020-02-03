from tqdm import tqdm
import json
from collections import defaultdict
import codecs

def evaluate(data):
    official_A, official_B, official_C = 1e-10, 1e-10, 1e-10

    results = []
    for d in tqdm(iter(data)):
        R = set([tuple(i) for i in d['predict']])
        official_T = set([tuple(i) for i in d['truth']])
        official_A += len(R & official_T)
        official_B += len(R)
        official_C += len(official_T)

    return 2 * official_A / (official_B + official_C), official_A / official_B, official_A / official_C



def split_file_by_overlapping_type(data):
    normal_results = []
    epo_results = []
    spo_results = []
    for d in tqdm(iter(data)):
        official_T = set([tuple(i) for i in d['truth']]) 
        head_dict = defaultdict(int)
        tail_dict = defaultdict(int)
        head_tail_dict = defaultdict(int)
        for spo in official_T:
            head_dict[spo[0]] += 1
            tail_dict[spo[2]] += 1
            head_tail_dict[(spo[0],spo[2])] += 1
        epo_flag = spo_flag = False
        for head_tail in head_tail_dict:
            if head_tail_dict[head_tail] > 1:
                epo_flag = True
        for head in head_dict:
            if head_dict[head] > 1:
                spo_flag = True
        for tail in tail_dict:
            if tail_dict[tail] > 1:
                spo_flag = True
        if epo_flag:
            epo_results.append(d)
        elif spo_flag:
            spo_results.append(d)
        if not spo_flag and not epo_flag:
            normal_results.append(d)
    return normal_results, spo_results, epo_results


def split_file_by_triplet_num(data):
    normal_results = []
    one_results = []
    two_results = []
    three_results = []
    four_results = []
    gfour_results = []
    for d in tqdm(iter(data)):
        official_T = set([tuple(i) for i in d['truth']]) 
        if len(official_T) == 1:
            one_results.append(d)
        elif len(official_T) == 2:
            two_results.append(d)
        elif len(official_T) == 3:
            three_results.append(d)
        elif len(official_T) == 4:
            four_results.append(d)
        elif len(official_T) > 4:
            gfour_results.append(d)        
    return one_results, two_results, three_results, four_results, gfour_results




if __name__ == '__main__':

    test_file_name = 'saved_models/multi/best_test_results.json' # You can obtain this file after running the eval.py script

    test_file = json.load(open(test_file_name))

    for res in split_by_length(test_file):
        print(evaluate(res))

