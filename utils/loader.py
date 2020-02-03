import json
import numpy as np
import random
from random import choice
from tqdm import tqdm
import collections

global num

def get_nearest_start_position(S1):
    nearest_start_list = []
    current_distance_list = []
    for start_pos_list in S1:
        nearest_start_pos = []
        current_start_pos = 0
        current_pos = []
        flag = False
        for i, start_label in enumerate(start_pos_list):
            if start_label > 0:
                current_start_pos = i
                flag = True
            nearest_start_pos.append(current_start_pos)
            if flag > 0:
                if i-current_start_pos > 10:
                    current_pos.append(499)
                else:
                    current_pos.append(i-current_start_pos)
            else:
                current_pos.append(499)
        # print(start_pos_list)
        # print(nearest_start_pos)
        # print(current_pos)
        # print('-----')
        nearest_start_list.append(nearest_start_pos)
        current_distance_list.append(current_pos)
    return nearest_start_list, current_distance_list

def locate_entity(token_list, entity):
    try:
        for i, token in enumerate(token_list):
            if entity.startswith(token):
                len_ = len(token)
                j = i+1 ; joined_tokens = [token]
                while len_ < len(entity):
                    len_ += len(token_list[j])
                    joined_tokens.append(token_list[j])
                    j = j+1
                if ''.join(joined_tokens) == entity:
                    return i, len(joined_tokens)
    except Exception:
        # print(entity,token_list)
        pass
    return -1, -1

def seq_padding(X):
    L = [len(x) for x in X]
    # ML =  config.MAX_LEN #max(L)
    ML = max(L)
    return [x + [0] * (ML - len(x)) for x in X]

def char_padding(X):
    L_S = [len(x) for x in X]
    ML_S = max(L_S)
    L = [[len(t) for t in s] for s in X]
    # ML =  config.MAX_LEN #max(L)
    ML = max([max(l) for l in L])
    if ML <= 15:
        return [[t + [0] * (15 - len(t)) for t in s]+[[1] * 15 for i in range(ML_S-len(s))] for s in X]
    else:
        return [[t + [0] * (15 - len(t)) if len(t) <=15 else t[:15] for t in s]+[[1] * 15 for i in range(ML_S-len(s))] for s in X]
def get_pos_tags(tokens, pos_tags, pos2id):
    pos_labels = [pos2id.get(flag,1) for flag in pos_tags]
    if len(pos_labels) != len(tokens):
        print(pos_labels)
        return False
    return pos_labels

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))


def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]



class DataLoader(object):
    def __init__(self, data, predicate2id, char2id, word2id, pos2id, subj_type2id, obj_type2id, batch_size=64, evaluation=False):
        
        self.batch_size = batch_size
        
        self.predicate2id = predicate2id
        self.char2id = char2id
        self.pos2id = pos2id
        self.subj_type2id = subj_type2id
        self.obj_type2id = obj_type2id
        self.word2id = word2id
        data = self.preprocess(data)
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.num_examples = len(data)
        self.data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]


    def preprocess(self, data):
        processed = []
        for d in data:
            # d = self.data[i]
            tokens = d['tokens']
            if len(tokens) > 150:
                continue
            items = {}
            subj_type = collections.defaultdict(list)
            obj_type = collections.defaultdict(list)
            for sp in d['spo_details']:
                key = (sp[0], sp[1])
                subj_type[key].append(self.subj_type2id[sp[2]])
                if key not in items:
                    items[key] = []
                items[key].append((sp[4],
                                    sp[5],
                                    self.predicate2id[sp[3]]))
                obj_type[(sp[4],sp[5])].append(self.obj_type2id[sp[6]])


                    
            if items:
                chars_ids = [[self.char2id.get(c, 1) for c in token] for token in tokens] # 1是unk，0是padding
                tokens_ids = [self.word2id.get(w, 1) for w in tokens]                 
                pos_tags = get_pos_tags(tokens, d['pos_tags'], self.pos2id)
                if not pos_tags:
                    continue
                s1, s2 = [0] * len(tokens), [0] * len(tokens)
                ts1, ts2 = [0] * len(tokens), [0] * len(tokens)

                for j in items:
                    s1[j[0]] = 1
                    s2[j[1]-1] = 1
                    stp = choice(subj_type[j])
                    ts1[j[0]] = stp
                    ts2[j[1]-1] = stp
                k1, k2 = choice(list(items.keys()))
                o1, o2 = [0] * len(tokens), [0] * len(tokens) 
                to1, to2 = [0] * len(tokens), [0] * len(tokens) 
                distance_to_subj = get_positions(k1, k2-1, len(tokens))

                for j in items[(k1, k2)]:
                    o1[j[0]] = j[2]
                    o2[j[1]-1] = j[2]
                    otp = choice(obj_type[(j[0], j[1])])
                    to1[j[0]] = otp
                    to2[j[1]-1] = otp
                processed += [(tokens_ids, chars_ids, [k1], [k2-1], s1, s2, o1, o2, ts1, ts2, to1, to2, pos_tags, distance_to_subj)]
        return processed

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 14
        lens = [len(x) for x in batch[0]]

        batch, orig_idx = sort_all(batch, lens)
        T = np.array(seq_padding(batch[0]))
        C = np.array(char_padding(batch[1]))
        K1, K2 = np.array(batch[2]), np.array(batch[3])
        
        
        S1 = np.array(seq_padding(batch[4]))
        S2 = np.array(seq_padding(batch[5]))
        O1 = np.array(seq_padding(batch[6]))
        O2 = np.array(seq_padding(batch[7]))
        TS1 = np.array(seq_padding(batch[8]))
        TS2 = np.array(seq_padding(batch[9]))
        TO1 = np.array(seq_padding(batch[10]))
        TO2 = np.array(seq_padding(batch[11]))
        Pos_tags = np.array(seq_padding(batch[12]))
        Distance_to_subj = np.array(seq_padding(batch[13]))
        Nearest_S1, Distance_S1 = get_nearest_start_position(batch[4])
        Nearest_S1, Distance_S1 = np.array(seq_padding(Nearest_S1)), np.array(seq_padding(Distance_S1))
        Nearest_O1, Distance_O1 = get_nearest_start_position(batch[6])
        Nearest_O1, Distance_O1 = np.array(seq_padding(Nearest_O1)), np.array(seq_padding(Distance_O1))


        return (T, C, Pos_tags, K1, K2, S1, S2, O1, O2, TS1, TS2, TO1, TO2, Nearest_S1, Distance_S1, Distance_to_subj, Nearest_O1, Distance_O1, orig_idx)


if __name__ == '__main__':
    s = [[[1,3,4],[2,2,2,2,2]],[[1,3,4,6,6,6,6,6,6],[2,2,2,2,2,7,7,7]]]
    print(char_padding(s))


