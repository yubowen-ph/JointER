"""
Prepare vocabulary, initial word vectors and tagging scheme.
"""
import json
import pickle
import argparse
import numpy as np
from collections import Counter, defaultdict


from utils import vocab, helper, constant

import codecs



def parse_args():
    parser = argparse.ArgumentParser(description='Prepare vocab for joint relation extraction.')
    parser.add_argument('--data_dir', default='dataset/NYT-multi/data', help='Input data directory.')
    parser.add_argument('--vocab_dir', default='dataset/NYT-multi/vocab', help='Output vocab directory.')
    parser.add_argument('--emb_dir', default='dataset/embedding', help='GloVe directory.')
    parser.add_argument('--wv_file', default='glove.840B.300d.txt', help='GloVe vector file.')
    parser.add_argument('--wv_dim', type=int, default=300, help='GloVe vector dimension.')
    parser.add_argument('--min_freq', type=int, default=0, help='If > 0, use min_freq as the cutoff.')
    parser.add_argument('--lower', action='store_true', help='If specified, lowercase all words.')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # input files
    train_file = args.data_dir + '/train.json'
    dev_file = args.data_dir + '/dev.json'
    test_file = args.data_dir + '/test.json'
    schema_file = args.data_dir + '/schemas.json'
    wv_file = args.emb_dir + '/' + args.wv_file
    wv_dim = args.wv_dim

    # output files
    helper.ensure_dir(args.vocab_dir)
    vocab_file = args.vocab_dir + '/vocab.pkl'
    char_file = args.vocab_dir + '/chars.json'
    emb_file = args.vocab_dir + '/embedding.npy'

    # load files
    print("loading files...")
    train_tokens = load_tokens(train_file)
    dev_tokens = load_tokens(dev_file)
    test_tokens = load_tokens(test_file)
    if args.lower:
        train_tokens, dev_tokens, test_tokens = [[t.lower() for t in tokens] for tokens in\
                (train_tokens, dev_tokens, test_tokens)]

    # load glove
    print("loading glove...")
    glove_vocab = vocab.load_glove_vocab(wv_file, wv_dim)
    print("{} words loaded from glove.".format(len(glove_vocab)))
    
    print("building vocab...")
    v = build_vocab(train_tokens, glove_vocab, args.min_freq)

    print("calculating oov...")
    datasets = {'train': train_tokens, 'dev': dev_tokens, 'test': test_tokens}
    for dname, d in datasets.items():
        total, oov = count_oov(d, v)
        print("{} oov: {}/{} ({:.2f}%)".format(dname, oov, total, oov*100.0/total))
    
    print("building embeddings...")
    embedding = vocab.build_embedding(wv_file, v, wv_dim)
    print("embedding size: {} x {}".format(*embedding.shape))

    print("dumping embeddings to files...")
    with open(vocab_file, 'wb') as outfile:
        pickle.dump(v, outfile)
    np.save(emb_file, embedding)
    # print("all done.")

    print("building schemas...")
    all_schemas = set()
    subj_type  = set()
    obj_type = set()
    min_count = 2
    pos_tags = set()
    chars = defaultdict(int)
    with open(train_file) as f:
        a = json.load(f)
        for ins in a:
            for spo in ins['spo_details']:
                all_schemas.add(spo[3])
                subj_type.add(spo[2])
                obj_type.add(spo[6])
            for pos in ins['pos_tags']:
                pos_tags.add(pos)
            for token in ins['tokens']:
                for char in token:
                    chars[char] += 1
    id2predicate = {i+1:j for i,j in enumerate(all_schemas)} # 0表示终止类别
    predicate2id = {j:i for i,j in id2predicate.items()}

    id2subj_type = {i+1:j for i,j in enumerate(subj_type)} # 0表示终止类别
    subj_type2id = {j:i for i,j in id2subj_type.items()}

    id2obj_type = {i+1:j for i,j in enumerate(obj_type)} # 0表示终止类别
    obj_type2id = {j:i for i,j in id2obj_type.items()}

    with codecs.open(schema_file, 'w', encoding='utf-8') as f:
        json.dump([id2predicate, predicate2id, id2subj_type, subj_type2id, id2obj_type, obj_type2id], f, indent=4, ensure_ascii=False)


    print("dumping chars to files...")
    with codecs.open(char_file, 'w', encoding='utf-8') as f:
        chars = {i:j for i,j in chars.items() if j >= min_count}
        id2char = {i+2:j for i,j in enumerate(chars)} # padding: 0, unk: 1
        char2id = {j:i for i,j in id2char.items()}
        id2pos = {i+2:j for i,j in enumerate(pos_tags)} # padding: 0, unk: 1
        pos2id = {j:i for i,j in id2pos.items()}
        json.dump([id2char, char2id, id2pos, pos2id], f, indent=4, ensure_ascii=False)






def load_tokens(filename):
    with open(filename) as infile:
        data = json.load(infile)
        tokens = []
        for d in data:
            # tokens.extend([word["word"] for word in d['postag']])
            tokens.extend(d['tokens'])
    print("{} tokens from {} examples loaded from {}.".format(len(tokens), len(data), filename))
    return tokens

def build_vocab(tokens, glove_vocab, min_freq):
    """ build vocab from tokens and glove words. """
    counter = Counter(t for t in tokens)
    # if min_freq > 0, use min_freq, otherwise keep all glove words
    if min_freq > 0:
        v = sorted([t for t in counter if counter.get(t) >= min_freq], key=counter.get, reverse=True)
    else:
        v = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)
    # add special tokens and entity mask tokens
    v = constant.VOCAB_PREFIX  + v
    print("vocab built with {}/{} words.".format(len(v), len(counter)))
    return v

def count_oov(tokens, vocab):
    c = Counter(t for t in tokens)
    total = sum(c.values())
    matched = sum(c[t] for t in vocab)
    return total, total-matched


if __name__ == '__main__':
    main()


