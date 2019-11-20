import numpy as np
from tqdm import tqdm
from utils import loader
import json

def extract_items(tokens_in, pos_tags, char2id, word2id, pos2id, id2predicate, model):
    R = []
    _t = [word2id.get(w, 1) for w in tokens_in] 
    _c = [[char2id.get(c, 1) for c in token] for token in tokens_in]
    _pos = loader.get_pos_tags(tokens_in, pos_tags, pos2id)

    _t =  np.array(loader.seq_padding([_t]))
    _c =  np.array(loader.char_padding([_c ]))
    _pos = np.array(loader.seq_padding([_pos]))
    _s1, _s2, hidden, sentence_rep = model.predict_subj_per_instance(_t, _c, _pos)
    _s1, _s2  = np.argmax(_s1, 1), np.argmax(_s2, 1)
    for i,_ss1 in enumerate(_s1):
        if _ss1 > 0:
            _subject = ''
            for j,_ss2 in enumerate(_s2[i:]):
                if _ss2 == _ss1:
                    _subject = ' '.join(tokens_in[i: i+j+1])
                    break
            if _subject:
                _k1, _k2 = np.array([[i]]), np.array([[i+j]])
                distance_to_subj = np.array([loader.get_positions(i, i+j, len(tokens_in))])
                _o1, _o2 = model.predict_obj_per_instance([_t,_k1, _k2, distance_to_subj], hidden, sentence_rep)
                _o1, _o2 = np.argmax(_o1, 1), np.argmax(_o2, 1)
                for i,_oo1 in enumerate(_o1):
                    if _oo1 > 0:
                        for j,_oo2 in enumerate(_o2[i:]):
                            if _oo2 == _oo1:
                                _object = ' '.join(tokens_in[i: i+j+1])
                                _predicate = id2predicate[_oo1]
                                R.append((_subject, _predicate, _object))
                                break
    return set(R)





def evaluate(data, char2id, word2id, pos2id, id2predicate, model):
    official_A, official_B, official_C = 1e-10, 1e-10, 1e-10
    manual_A, manual_B, manual_C = 1e-10, 1e-10, 1e-10

    results = []
    for d in tqdm(iter(data)):
        R = extract_items(d['tokens'],d['pos_tags'], char2id, word2id, pos2id, id2predicate, model)
        official_T = set([tuple(i) for i in d['spo_list']])
        manual_T = set([tuple(i) for i in d['spo_manual']])
        results.append({'text':' '.join(d['tokens']), 'predict':list(R), 'truth':list(official_T), 'maunal':list(manual_T)})
        official_A += len(R & official_T)
        official_B += len(R)
        official_C += len(official_T)
        manual_A += len(R & manual_T)
        manual_B += len(R)
        manual_C += len(manual_T)
    return 2 * official_A / (official_B + official_C), official_A / official_B, official_A / official_C, 2 * manual_A / (manual_B + manual_C), manual_A / manual_B, manual_A / manual_C, results









def test(data, char2id, word2id, pos2id, id2predicate, model):
    with open('result.json','w') as fw:
        for d in tqdm(iter(data)):
            R = set(extract_items(d['text'],d['tokens_jieba'], d['pos_tags'], char2id, word2id, pos2id, id2predicate, model))
            a = {}
            a['text'] = d['text']
            a['spo_list'] = [{"object_type":"XX","subject_type":"XX","subject":c[0],"predicate":c[1],"object":c[2]} for c in R]
            fw.write(json.dumps(a, ensure_ascii=False)+'\n')



    


