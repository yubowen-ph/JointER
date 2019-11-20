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


parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='data/dataset')
parser.add_argument('--dataset', type=str, default='test_me', help="Evaluate on dev or test.")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(1234)
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
batch = json.load(open(data_file))
id2predicate, predicate2id = json.load(open(args.data_dir + '/all_50_schemas_me.json'))
id2predicate = {int(i):j for i,j in id2predicate.items()}
id2char, char2id = json.load(open(opt['vocab_dir'] + '/all_chars_me.json'))

helper.print_config(opt)

dev_f1, dev_p, dev_r = score.evaluate(batch, char2id, id2predicate, model)

print("data_file: {}: p = {:.6f}, r = {:.6f}, f1 = {:.4f}".format(args.dataset, dev_p, dev_r, dev_f1))
print("Evaluation ended.")

