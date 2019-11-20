"""
A rnn model for relation extraction, written in pytorch.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
from utils import torch_utils, loader
from models import layers, submodel


class RelationModel(object):
    """ A wrapper class for the training and evaluation of models. """
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.model = BiLSTMCNN(opt, emb_matrix)
        self.subj_criterion = nn.BCELoss(reduction='none')
        self.obj_criterion = nn.CrossEntropyLoss(reduction='none')
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.subj_criterion.cuda()
            self.obj_criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'], opt['weight_decay'])
    

    def update(self, batch):
        """ Run a step of forward and backward model update. """
        if self.opt['cuda']:
            inputs = [Variable(torch.LongTensor(b).cuda()) for b in batch[:5]]
            subj_start_binary = Variable(torch.LongTensor(batch[5]).cuda()).float()
            subj_end_binary = Variable(torch.LongTensor(batch[6]).cuda()).float()
            obj_start_relation = Variable(torch.LongTensor(batch[7]).cuda())
            obj_end_relation = Variable(torch.LongTensor(batch[8]).cuda())
            subj_start_type = Variable(torch.LongTensor(batch[9]).cuda())
            subj_end_type = Variable(torch.LongTensor(batch[10]).cuda())
            obj_start_type = Variable(torch.LongTensor(batch[11]).cuda())
            obj_end_type = Variable(torch.LongTensor(batch[12]).cuda())
            nearest_subj_start_position_for_each_token = Variable(torch.LongTensor(batch[13]).cuda())
            distance_to_nearest_subj_start = Variable(torch.LongTensor(batch[14]).cuda())
            distance_to_subj = Variable(torch.LongTensor(batch[15]).cuda())
            nearest_obj_start_position_for_each_token = Variable(torch.LongTensor(batch[16]).cuda())
            distance_to_nearest_obj_start = Variable(torch.LongTensor(batch[17]).cuda())
        else:
            inputs = [Variable(torch.LongTensor(b)) for b in batch[:4]]
            subj_start_label = Variable(torch.LongTensor(batch[4])).float()
            subj_end_label = Variable(torch.LongTensor(batch[5])).float()
            obj_start_label = Variable(torch.LongTensor(batch[6]))
            obj_end_label = Variable(torch.LongTensor(batch[7]))
            subj_type_start_label = Variable(torch.LongTensor(batch[8]))
            subj_type_end_label = Variable(torch.LongTensor(batch[9]))
            obj_type_start_label = Variable(torch.LongTensor(batch[10]))
            obj_type_end_label = Variable(torch.LongTensor(batch[11]))
            subj_nearest_start_for_each = Variable(torch.LongTensor(batch[12]))
            subj_distance_to_start = Variable(torch.LongTensor(batch[13]))
        
        
        mask = (inputs[0].data>0).float()
        # step forward
        self.model.train()
        self.optimizer.zero_grad()


        
        
        subj_start_logits, subj_end_logits, obj_start_logits, obj_end_logits = self.model(inputs, mask, nearest_subj_start_position_for_each_token, distance_to_nearest_subj_start, distance_to_subj, nearest_obj_start_position_for_each_token, distance_to_nearest_obj_start)

        
        subj_start_loss = self.obj_criterion(subj_start_logits.view(-1, self.opt['num_subj_type']+1), subj_start_type.view(-1).squeeze()).view_as(mask)
        subj_start_loss = torch.sum(subj_start_loss.mul(mask.float()))/torch.sum(mask.float())
        
        subj_end_loss = self.obj_criterion(subj_end_logits.view(-1, self.opt['num_subj_type']+1), subj_end_type.view(-1).squeeze()).view_as(mask)
        subj_end_loss = torch.sum(subj_end_loss.mul(mask.float()))/torch.sum(mask.float())
        
        obj_start_loss = self.obj_criterion(obj_start_logits.view(-1, self.opt['num_class']+1), obj_start_relation.view(-1).squeeze()).view_as(mask)
        obj_start_loss = torch.sum(obj_start_loss.mul(mask.float()))/torch.sum(mask.float())
        
        obj_end_loss = self.obj_criterion(obj_end_logits.view(-1, self.opt['num_class']+1), obj_end_relation.view(-1).squeeze()).view_as(mask)
        obj_end_loss = torch.sum(obj_end_loss.mul(mask.float()))/torch.sum(mask.float())
        
        loss = self.opt['subj_loss_weight']*(subj_start_loss + subj_end_loss) + (obj_start_loss + obj_end_loss)
        
        # backward
        loss.backward()
        # torch.nn.utils.clip_grad_norm(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.data.item()
        return loss_val



    def predict_subj_per_instance(self, words, chars, pos_tags):
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """
        if self.opt['cuda']:
            words = Variable(torch.LongTensor(words).cuda())
            chars = Variable(torch.LongTensor(chars).cuda())
            pos_tags = Variable(torch.LongTensor(pos_tags).cuda())
        else:
            words = Variable(torch.LongTensor(words))
            features = Variable(torch.LongTensor(features))

        batch_size, seq_len = words.size()
        mask = (words.data>0).float()
        # forward
        self.model.eval()
        inputs, hidden, sentence_rep = self.model.based_encoder(words, chars, pos_tags, mask)

        subj_start_logits, subj_start_outputs = self.model.subj_sublayer.predict_subj_start(hidden, sentence_rep, mask)

        _s1 = np.argmax(subj_start_logits, 1)
        
        nearest_subj_position_for_each_token, distance_to_nearest_subj =  loader.get_nearest_start_position([_s1])
        nearest_subj_position_for_each_token, distance_to_nearest_subj = Variable(torch.LongTensor(np.array(nearest_subj_position_for_each_token)).cuda()), Variable(torch.LongTensor(np.array(distance_to_nearest_subj)).cuda())

        subj_end_logits = self.model.subj_sublayer.predict_subj_end(subj_start_outputs, mask, nearest_subj_position_for_each_token, distance_to_nearest_subj, sentence_rep)
        
        return subj_start_logits, subj_end_logits, hidden, sentence_rep

    def predict_obj_per_instance(self, inputs, hidden, sentence_rep):
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """
        if self.opt['cuda']:
            inputs = [Variable(torch.LongTensor(b).cuda()) for b in inputs]
        else:
            inputs = [Variable(torch.LongTensor(b)).unsqueeze(0) for b in inputs[:4]]
        mask = (inputs[0].data>0).float()

        words, subj_start_position, subj_end_position, distance_to_subj = inputs # unpack

        self.model.eval()

        obj_start_logits, obj_start_outputs = self.model.obj_sublayer.predict_obj_start(hidden, sentence_rep, subj_start_position, subj_end_position, mask, distance_to_subj)

        _o1 = np.argmax(obj_start_logits, 1)
        nearest_obj_start_position_for_each_token, distance_to_nearest_obj_start =  loader.get_nearest_start_position([_o1])
        nearest_obj_start_position_for_each_token, distance_to_nearest_obj_start = Variable(torch.LongTensor(np.array(nearest_obj_start_position_for_each_token)).cuda()), Variable(torch.LongTensor(np.array(distance_to_nearest_obj_start)).cuda())


        obj_end_logits = self.model.obj_sublayer.predict_obj_end(obj_start_outputs, mask, nearest_obj_start_position_for_each_token, distance_to_nearest_obj_start)

         
        return obj_start_logits, obj_end_logits






    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                'epoch': epoch
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

class BiLSTMCNN(nn.Module):
    """ A sequence model for relation extraction. """

    def __init__(self, opt, emb_matrix=None):
        super(BiLSTMCNN, self).__init__()
        self.drop = nn.Dropout(opt['dropout'])
        self.word_emb = nn.Embedding(opt['word_vocab_size'], opt['word_emb_dim'], padding_idx=0)
        self.char_emb = nn.Embedding(opt['char_vocab_size'], opt['char_emb_dim'], padding_idx=0)
        self.pos_emb = nn.Embedding(opt['pos_size'], opt['pos_emb_dim'], padding_idx=0)
        self.input_size = opt['word_emb_dim']+opt['char_hidden_dim']+opt['pos_emb_dim']
        self.rnn = nn.LSTM(self.input_size, opt['hidden_dim'], opt['num_layers'], batch_first=True,\
                dropout=opt['dropout'], bidirectional=True)
        self.subj_sublayer = submodel.SubjTypeModel(opt, 4*opt['hidden_dim'], 2*opt['hidden_dim'])
        self.obj_sublayer = submodel.ObjBaseModel(opt, 8*opt['hidden_dim'], 2*opt['hidden_dim'])
        self.char_encoder = layers.CharEncoder(opt)
        self.opt = opt
        self.topn = self.opt.get('topn', 1e10)
        self.use_cuda = opt['cuda']
        self.emb_matrix = emb_matrix
        self.init_weights()
    
    def init_weights(self):
        if self.emb_matrix is None:
            self.word_emb.weight.data[1:,:].uniform_(-1.0, 1.0) # keep padding dimension to be 0
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.word_emb.weight.data.copy_(self.emb_matrix)
        self.char_emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        self.pos_emb.weight.data[1:,:].uniform_(-1.0, 1.0)


  
        # decide finetuning
        if self.topn <= 0:
            print("Do not finetune word embedding layer.")
            self.word_emb.weight.requires_grad = False
        elif self.topn < self.opt['word_vocab_size']:
            print("Finetune top {} word embeddings.".format(self.topn))
            self.word_emb.weight.register_hook(lambda x: \
                    torch_utils.keep_partial_grad(x, self.topn))
        else:
            print("Finetune all embeddings.")

    def zero_state(self, batch_size): 
        state_shape = (2*self.opt['num_layers'], batch_size, self.opt['hidden_dim'])
        h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
        if self.use_cuda:
            return h0.cuda(), c0.cuda()
        else:
            return h0, c0

    def based_encoder(self, words, chars, pos_tags, mask):
        if mask.shape[0] == 1:
            seq_lens = [mask.sum(1).squeeze()]
        else:
            seq_lens = list(mask.sum(1).squeeze())
        
        batch_size,seq_len = words.size()
        
        # embedding lookup
        word_inputs = self.drop(self.word_emb(words))
        pos_inputs = self.pos_emb(pos_tags)
        chars = chars.view(-1,15)
        chars_mask = (chars.data>0).float()
        char_inputs = self.char_encoder(self.drop(self.char_emb(chars)), chars_mask).contiguous().view(batch_size,seq_len,self.opt['char_hidden_dim'])
        # highway_output = self.highway(torch.cat([word_inputs,char_inputs],dim=2))
        inputs = [word_inputs,char_inputs, pos_inputs]
        inputs = self.drop(torch.cat(inputs, dim=2))
        h0, c0 = self.zero_state(batch_size)
        packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs, seq_lens, batch_first=True)
        hidden, (ht, ct) = self.rnn(packed_inputs, (h0, c0))
        hidden, output_lens = nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True)

        
        hidden_masked = hidden - ((1 - mask) * 1e10).unsqueeze(2).repeat(1,1,hidden.shape[2]).float()
        sentence_rep = F.max_pool1d(torch.transpose(hidden_masked, 1, 2), hidden_masked.size(1)).squeeze(2)
        return inputs, hidden, sentence_rep
    
    def forward(self, inputs, mask, nearest_subj_position_for_each_token, distance_to_nearest_subj, distance_to_subj, nearest_obj_start_position_for_each_token, distance_to_nearest_obj_start):

        words, chars, pos_tags, subj_start_position, subj_end_position = inputs # unpack
        batch_size, seq_len = words.size()
        inputs, hidden, sentence_rep = self.based_encoder(words, chars, pos_tags, mask)
        subj_start_logits, subj_end_logits = self.subj_sublayer(hidden, sentence_rep, mask, nearest_subj_position_for_each_token, distance_to_nearest_subj)
        obj_start_logits, obj_end_logits = self.obj_sublayer(hidden, sentence_rep, subj_start_position, subj_end_position, mask, distance_to_subj, nearest_obj_start_position_for_each_token, distance_to_nearest_obj_start)

        
        return subj_start_logits, subj_end_logits, obj_start_logits, obj_end_logits
    

