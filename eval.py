"""
Run evaluation with saved models.
"""

import os
import random
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import json
from models.model import RelationModel
from utils import torch_utils, helper, score
from utils.vocab import Vocab



parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/NYT-multi/data')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
model = RelationModel(opt)
model.load(model_file)



# load data
data_file = args.data_dir + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
data = json.load(open(data_file))
id2predicate, predicate2id, id2subj_type, subj_type2id, id2obj_type, obj_type2id = json.load(open(opt['data_dir'] + '/schemas.json'))
id2predicate = {int(i):j for i,j in id2predicate.items()}
id2char, char2id, id2pos, pos2id = json.load(open(opt['vocab_dir'] + '/chars.json'))
vocab_file = opt['vocab_dir'] + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
word2id = vocab.word2id



helper.print_config(opt)

f1, p, r, results = score.evaluate(data, char2id, word2id, pos2id, id2predicate, model)
results_save_dir = opt['model_save_dir'] + '/best_{}_results.json'.format(args.dataset)
print("Dumping the best test results to {}".format(results_save_dir))
with open(results_save_dir, 'w') as fw:
    json.dump(results, fw, indent=4, ensure_ascii=False)

print("data_file: {}: p = {:.6f}, r = {:.6f}, f1 = {:.4f}".format(args.dataset, p, r, f1))
print("Evaluation ended.")

