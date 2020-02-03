import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from utils import torch_utils

from models.layers import *



class SubjTypeModel(nn.Module):

    
    def __init__(self, opt, input_size, hidden_dim, filter=3):
        super(SubjTypeModel, self).__init__()
        self.input_size = input_size
        self.dropout = nn.Dropout(opt['dropout'])
        self.hidden_dim = hidden_dim
        self.position_embedding = nn.Embedding(500, opt['position_emb_dim'])
        self.rnn_start = nn.LSTM(self.input_size, hidden_dim//2, 1, batch_first=True, bidirectional=True)
        self.rnn_end = nn.LSTM(self.input_size+opt['position_emb_dim'], hidden_dim//2, 1, batch_first=True, bidirectional=True)
        self.linear_subj_start = nn.Linear(hidden_dim, opt['num_subj_type']+1)
        self.linear_subj_end = nn.Linear(hidden_dim, opt['num_subj_type']+1)
        self.init_weights()


    def zero_state(self, batch_size): 
        state_shape = (2, batch_size, self.hidden_dim//2)
        h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
        return h0.cuda(), c0.cuda()


    def init_weights(self):

        self.position_embedding.weight.data.uniform_(-1.0, 1.0)
        self.linear_subj_start.bias.data.fill_(0)
        init.xavier_uniform_(self.linear_subj_start.weight, gain=1) # initialize linear layer

        self.linear_subj_end.bias.data.fill_(0)
        init.xavier_uniform_(self.linear_subj_end.weight, gain=1) # initialize linear layer




    def forward(self, hidden, sentence_rep, masks, nearest_subj_position_for_each_token, distance_to_nearest_subj): #


        batch_size, seq_len, input_size = hidden.size()

        mask = masks.long()
        if mask.shape[0] == 1:
            seq_lens = [mask.sum(1).squeeze()]
        else:
            seq_lens = list(mask.sum(1).squeeze())
            
        distance_to_nearest_subj_emb = self.position_embedding(distance_to_nearest_subj)
        subj_inputs = torch.cat([hidden, seq_and_vec(seq_len,sentence_rep)], dim=2)
        subj_start_inputs = self.dropout(subj_inputs)

        h0, c0 = self.zero_state(batch_size)
        subj_start_outputs = nn.utils.rnn.pack_padded_sequence(subj_start_inputs, seq_lens, batch_first=True)
        subj_start_outputs, (ht, ct) = self.rnn_start(subj_start_outputs, (h0, c0))
        subj_start_outputs, output_lens = nn.utils.rnn.pad_packed_sequence(subj_start_outputs, batch_first=True)
        subj_end_inputs = torch.cat([subj_start_outputs, distance_to_nearest_subj_emb, seq_and_vec(seq_len,sentence_rep)], dim=2)
        
        subj_end_inputs = self.dropout(subj_end_inputs)
        subj_start_outputs = self.dropout(subj_start_outputs)
        subj_start_logits = self.linear_subj_start(subj_start_outputs)

        
        subj_end_outputs = nn.utils.rnn.pack_padded_sequence(subj_end_inputs, seq_lens, batch_first=True)
        subj_end_outputs, (ht, ct) = self.rnn_end(subj_end_outputs, (h0, c0))
        subj_end_outputs, output_lens = nn.utils.rnn.pad_packed_sequence(subj_end_outputs, batch_first=True)
        
        subj_end_outputs = self.dropout(subj_end_outputs)

        subj_end_logits = self.linear_subj_end(subj_end_outputs)

        return subj_start_logits.squeeze(-1), subj_end_logits.squeeze(-1)


    def predict_subj_start(self, hidden, sentence_rep, masks):


        batch_size, seq_len, input_size = hidden.size()

        mask = masks.long()
        if mask.shape[0] == 1:
            seq_lens = [mask.sum(1).squeeze()]
        else:
            seq_lens = list(mask.sum(1).squeeze())
            
        subj_start_inputs = torch.cat([hidden, seq_and_vec(seq_len,sentence_rep)], dim=2)
        h0, c0 = self.zero_state(batch_size)
        subj_start_outputs = nn.utils.rnn.pack_padded_sequence(subj_start_inputs, seq_lens, batch_first=True)
        subj_start_outputs, (ht, ct) = self.rnn_start(subj_start_outputs, (h0, c0))
        subj_start_outputs, output_lens = nn.utils.rnn.pad_packed_sequence(subj_start_outputs, batch_first=True)
        subj_start_logits = self.linear_subj_start(subj_start_outputs)

        return subj_start_logits.squeeze(-1)[0].data.cpu().numpy(), subj_start_outputs



    def predict_subj_end(self, subj_start_outputs, masks, nearest_subj_position_for_each_token, distance_to_nearest_subj, sentence_rep):


        batch_size, seq_len, input_size = subj_start_outputs.size()

        mask = masks.long()
        if mask.shape[0] == 1:
            seq_lens = [mask.sum(1).squeeze()]
        else:
            seq_lens = list(mask.sum(1).squeeze())
            

        distance_to_nearest_subj_emb = self.position_embedding(distance_to_nearest_subj)
        h0, c0 = self.zero_state(batch_size)
        subj_end_inputs = torch.cat([subj_start_outputs, distance_to_nearest_subj_emb, seq_and_vec(seq_len,sentence_rep)], dim=2)
        subj_end_outputs = nn.utils.rnn.pack_padded_sequence(subj_end_inputs, seq_lens, batch_first=True)
        subj_end_outputs, (ht, ct) = self.rnn_end(subj_end_outputs, (h0, c0))
        subj_end_outputs, output_lens = nn.utils.rnn.pad_packed_sequence(subj_end_outputs, batch_first=True)

        subj_end_logits = self.linear_subj_end(subj_end_outputs)

        return subj_end_logits.squeeze(-1)[0].data.cpu().numpy()





class ObjBaseModel(nn.Module):

    
    def __init__(self, opt, input_size, hidden_dim, filter=3):
        super(ObjBaseModel, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim

        self.dropout = self.drop = nn.Dropout(opt['dropout'])
        self.rnn_start = nn.LSTM(self.input_size+opt['position_emb_dim'], hidden_dim//2, 1, batch_first=True, bidirectional=True)
        self.rnn_end = nn.LSTM(self.input_size+2*opt['position_emb_dim'], hidden_dim//2, 1, batch_first=True, bidirectional=True)
        self.distance_to_subj_embedding = nn.Embedding(400, opt['position_emb_dim'])
        self.distance_to_obj_start_embedding = nn.Embedding(500, opt['position_emb_dim'])
        self.linear_obj_start = nn.Linear(hidden_dim, opt['num_class']+1)
        self.linear_obj_end = nn.Linear(hidden_dim, opt['num_class']+1)
        self.init_weights()

    def init_weights(self):
        self.linear_obj_start.bias.data.fill_(0)
        init.xavier_uniform_(self.linear_obj_start.weight, gain=1) # initialize linear layer
        self.linear_obj_end.bias.data.fill_(0)
        init.xavier_uniform_(self.linear_obj_end.weight, gain=1) # initialize linear layer
        self.distance_to_subj_embedding.weight.data.uniform_(-1.0, 1.0)
        self.distance_to_obj_start_embedding.weight.data.uniform_(-1.0, 1.0)

    def zero_state(self, batch_size): 
        state_shape = (2, batch_size, self.hidden_dim//2)
        h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
        return h0.cuda(), c0.cuda()


    def forward(self, hidden, sentence_rep, subj_start_position, subj_end_position, masks, distance_to_subj, nearest_obj_start_position_for_each_token, distance_to_nearest_obj_start):


        mask = masks.long()
        if mask.shape[0] == 1:
            seq_lens = [mask.sum(1).squeeze()]
        else:
            seq_lens = list(mask.sum(1).squeeze())
        batch_size, seq_len, input_size = hidden.shape
        subj_start_hidden = torch.gather(hidden, dim=1, index=subj_start_position.unsqueeze(2).repeat(1,1,input_size)).squeeze(1)       
        subj_end_hidden = torch.gather(hidden, dim=1, index=subj_end_position.unsqueeze(2).repeat(1,1,input_size)).squeeze(1)       
        distance_to_subj_emb = self.distance_to_subj_embedding(distance_to_subj+200)   # To avoid negative indices     
        subj_related_info = torch.cat([seq_and_vec(seq_len,sentence_rep), seq_and_vec(seq_len,subj_start_hidden), seq_and_vec(seq_len,subj_end_hidden), distance_to_subj_emb], dim=2)
        obj_inputs = torch.cat([hidden, subj_related_info], dim=2)    
        obj_start_inputs = self.dropout(obj_inputs)


        distance_to_nearest_obj_emb = self.distance_to_obj_start_embedding(distance_to_nearest_obj_start)

        h0, c0 = self.zero_state(batch_size)
        obj_start_outputs = nn.utils.rnn.pack_padded_sequence(obj_start_inputs, seq_lens, batch_first=True)
        obj_start_outputs, (ht, ct) = self.rnn_start(obj_start_outputs, (h0, c0))
        obj_start_outputs, output_lens = nn.utils.rnn.pad_packed_sequence(obj_start_outputs, batch_first=True)

        subj_end_inputs = torch.cat([obj_start_outputs, subj_related_info, distance_to_nearest_obj_emb], dim=2)

        obj_end_inputs = self.dropout(subj_end_inputs)
        obj_start_outputs = self.dropout(obj_start_outputs)
        obj_start_logits = self.linear_obj_start(obj_start_outputs)
        
        obj_end_outputs = nn.utils.rnn.pack_padded_sequence(obj_end_inputs, seq_lens, batch_first=True)
        obj_end_outputs, (ht, ct) = self.rnn_end(obj_end_outputs, (h0, c0))
        obj_end_outputs, output_lens = nn.utils.rnn.pad_packed_sequence(obj_end_outputs, batch_first=True)
        obj_end_outputs = self.dropout(obj_end_outputs)

        obj_end_logits = self.linear_obj_end(obj_end_outputs)
        return obj_start_logits, obj_end_logits

    def predict_obj_start(self, hidden, sentence_rep, subj_start_position, subj_end_position, masks, distance_to_subj):


        batch_size, seq_len, input_size = hidden.size()

        mask = masks.long()
        if mask.shape[0] == 1:
            seq_lens = [mask.sum(1).squeeze()]
        else:
            seq_lens = list(mask.sum(1).squeeze())

        subj_start_hidden = torch.gather(hidden, dim=1, index=subj_start_position.unsqueeze(2).repeat(1,1,input_size)).squeeze(1)       
        subj_end_hidden = torch.gather(hidden, dim=1, index=subj_end_position.unsqueeze(2).repeat(1,1,input_size)).squeeze(1)       
        distance_to_subj_emb = self.distance_to_subj_embedding(distance_to_subj+200)        
        subj_related_info = torch.cat([seq_and_vec(seq_len,sentence_rep), seq_and_vec(seq_len,subj_start_hidden), seq_and_vec(seq_len,subj_end_hidden), distance_to_subj_emb], dim=2)
        obj_inputs = torch.cat([hidden, subj_related_info], dim=2)    
            
        h0, c0 = self.zero_state(batch_size)
        obj_start_outputs = nn.utils.rnn.pack_padded_sequence(obj_inputs, seq_lens, batch_first=True)
        obj_start_outputs, (ht, ct) = self.rnn_start(obj_start_outputs, (h0, c0))
        obj_start_outputs, output_lens = nn.utils.rnn.pad_packed_sequence(obj_start_outputs, batch_first=True)
        obj_start_logits = self.linear_obj_start(obj_start_outputs)

        obj_start_outputs = torch.cat([obj_start_outputs, subj_related_info], dim=2)
        return obj_start_logits.squeeze(-1)[0].data.cpu().numpy(), obj_start_outputs



    def predict_obj_end(self, obj_start_outputs, masks, nearest_obj_start_position_for_each_token, distance_to_nearest_obj_start):


        batch_size, seq_len, input_size = obj_start_outputs.size()

        mask = masks.long()
        if mask.shape[0] == 1:
            seq_lens = [mask.sum(1).squeeze()]
        else:
            seq_lens = list(mask.sum(1).squeeze())
            

        distance_to_nearest_obj_emb = self.distance_to_obj_start_embedding(distance_to_nearest_obj_start)
        obj_end_inputs = torch.cat([obj_start_outputs, distance_to_nearest_obj_emb], dim=2)

        h0, c0 = self.zero_state(batch_size)
        obj_end_outputs = nn.utils.rnn.pack_padded_sequence(obj_end_inputs, seq_lens, batch_first=True)
        obj_end_outputs, (ht, ct) = self.rnn_end(obj_end_outputs, (h0, c0))
        obj_end_outputs, output_lens = nn.utils.rnn.pad_packed_sequence(obj_end_outputs, batch_first=True)

        obj_end_logits = self.linear_obj_end(obj_end_outputs)

        return obj_end_logits.squeeze(-1)[0].data.cpu().numpy()