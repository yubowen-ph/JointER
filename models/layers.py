import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import math

from utils import torch_utils


class CharEncoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.char_encoder = nn.Conv1d(in_channels=opt['char_emb_dim'], out_channels=opt['char_hidden_dim'], kernel_size=3,
                                            padding=1)
        nn.init.xavier_uniform_(self.char_encoder.weight)
        nn.init.uniform_(self.char_encoder.bias)

    def forward(self, x, mask):
        x = self.char_encoder(x.transpose(1,2)).transpose(1,2)
        x = F.relu(x)
        hidden_masked = x - ((1 - mask) * 1e10).unsqueeze(2).repeat(1,1,x.shape[2]).float()
        sentence_rep = F.max_pool1d(torch.transpose(hidden_masked, 1, 2), hidden_masked.size(1)).squeeze(2)
        return sentence_rep






def seq_and_vec(seq_len, vec):
    return vec.unsqueeze(1).repeat(1,seq_len,1)


