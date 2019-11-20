import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import math

from utils import torch_utils

d_model = 128
dropout = 0.1
n_head = 4
d_k = d_model // n_head




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


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, dim=1, bias=True):
        super().__init__()
        if dim == 1:
            self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        else:
            raise Exception("Wrong dimension for Depthwise Separable Convolution!")
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.depthwise_conv.bias, 0.0)
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.pointwise_conv.bias, 0.0)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))

class PosEncoder(nn.Module):
    def __init__(self, length):
        super().__init__()
        freqs = torch.Tensor(
            [10000 ** (-i / d_model) if i % 2 == 0 else -10000 ** ((1 - i) / d_model) for i in range(d_model)]).unsqueeze(dim=1)
        phases = torch.Tensor([0 if i % 2 == 0 else math.pi / 2 for i in range(d_model)]).unsqueeze(dim=1)
        pos = torch.arange(length).repeat(d_model, 1).to(torch.float)
        self.pos_encoding = nn.Parameter(torch.sin(torch.add(torch.mul(pos, freqs), phases)), requires_grad=False)

    def forward(self, x):
        x = x + self.pos_encoding
        return x


class GLDR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(GLDR, self).__init__()
        self.dimensionality_reduction_block = DepthwiseSeparableConv(in_channels,out_channels,3)
        self.residual_block = nn.ModuleList([GCNN1d(out_channels, out_channels, 3, padding=diala, dilation=diala) for diala in dilation])
        self.dropout = nn.Dropout(0.1)
        # self.refinement_block = nn.ModuleList([GCNN1d(out_channels, out_channels, 3, padding=diala, dilation=diala) for diala in dilation])
    def forward(self, x):
        x = self.dimensionality_reduction_block(x)
        for residual_model in self.residual_block:
            x = self.dropout(x)
            x = residual_model(x)
        return x

class HighwayLayer(nn.Module):
    # TODO: We may need to add weight decay here
    def __init__(self, size, bias_init=0.0, nonlin=nn.ReLU(inplace=True), gate_nonlin=F.sigmoid):
        super(HighwayLayer, self).__init__()

        self.nonlin = nonlin
        self.gate_nonlin = gate_nonlin
        self.lin = nn.Linear(size, size)
        self.gate_lin = nn.Linear(size, size)
        self.gate_lin.bias.data.fill_(bias_init)
        self.lin.bias.data.fill_(bias_init)
        init.xavier_uniform_(self.gate_lin.weight, gain=1) # initialize linear layer
        init.xavier_uniform_(self.lin.weight, gain=1) # initialize linear layer


    def forward(self, x):
        trans = self.nonlin(self.lin(x))
        gate = self.gate_nonlin(self.gate_lin(x))
        return torch.add(torch.mul(gate, trans), torch.mul((1 - gate), x))


class HighwayNet(nn.Module):
    def __init__(self, depth, size):
        super(HighwayNet, self).__init__()

        layers = [HighwayLayer(size) for _ in range(depth)]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class GCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(GCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_gate = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        # self.trans = nn.Linear(kernel_size[1], out_channels)
        self.init_weights()

    def init_weights(self): 
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.uniform_(self.conv.bias)
        nn.init.xavier_uniform_(self.conv_gate.weight)
        nn.init.uniform_(self.conv_gate.bias)
        # self.trans.bias.data.fill_(0)
        # init.xavier_uniform_(self.trans.weight, gain=1) # initialize linear layer

    def forward(self, inputs):
        # print(self.conv_gate(inputs).shape)
        gate = torch.sigmoid(self.conv_gate(inputs))
        gated_outputs = torch.transpose(torch.mul(gate, self.conv(inputs)).squeeze(3), 1, 2)
        # outputs = gated_outputs + self.trans(inputs.squeeze(1))
        outputs = gated_outputs + inputs.squeeze(1)
        
        return outputs


class GCNN1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1):
        super(GCNN1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.conv_gate = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.init_weights()

    def init_weights(self): 
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.uniform_(self.conv.bias)
        nn.init.xavier_uniform_(self.conv_gate.weight)
        nn.init.uniform_(self.conv_gate.bias)


    def forward(self, inputs):
        gate = torch.sigmoid(self.conv_gate(inputs))
        gated_outputs = torch.mul(gate, self.conv(inputs))
        # outputs = gated_outputs + self.trans(inputs.squeeze(1))
        outputs = gated_outputs + inputs
        
        return outputs   

class EncoderBlock(nn.Module):
    def __init__(self, conv_num: int, ch_num: int, k: int, length: int):
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(ch_num, ch_num, k) for _ in range(conv_num)])
        self.self_att = MultiHeadAttention(d_model,n_head)
        self.fc = nn.Linear(ch_num, ch_num, bias=True)
        self.pos = PosEncoder(length)
        # self.norm = nn.LayerNorm([d_model, length])
        self.normb = nn.LayerNorm([d_model, length])
        self.norms = nn.ModuleList([nn.LayerNorm([d_model, length]) for _ in range(conv_num)])
        self.norme = nn.LayerNorm([d_model, length])
        self.L = conv_num
        self.init_weights()
    
    def init_weights(self):
        self.fc.bias.data.fill_(0)
        init.xavier_uniform_(self.fc.weight, gain=1) # initialize linear layer

    def forward(self, x, mask):
        out = self.pos(x)
        # out = x
        res = out
        out = self.normb(out)
        for i, conv in enumerate(self.convs):
            out = conv(out)
            out = F.relu(out)
            out = out + res
            if (i + 1) % 2 == 0:
                p_drop = dropout * (i + 1) / self.L
                out = F.dropout(out, p=p_drop, training=self.training)
            res = out
            out = self.norms[i](out)
        # print("Before attention: {}".format(out.size()))
        out = self.self_att(out, mask)
        # print("After attention: {}".format(out.size()))
        out = out + res
        out = F.dropout(out, p=dropout, training=self.training)
        res = out
        out = self.norme(out)
        out = self.fc(out.transpose(1, 2)).transpose(1, 2)
        out = F.relu(out)
        out = out + res
        out = F.dropout(out, p=dropout, training=self.training)
        return out




class MultiHeadAttention(nn.Module):
    def __init__(self, d_model_, n_head_):
        super().__init__()

        self.d_model = d_model_
        self.n_head = n_head_
        self.d_k = d_model_ // n_head_

        self.q_linear = nn.Linear(self.d_model, self.d_model)
        self.v_linear = nn.Linear(self.d_model, self.d_model)
        self.k_linear = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.d_model, self.d_model)
        self.a = 1 / math.sqrt(self.d_k)
        self.init_weights()

    def init_weights(self):
        self.q_linear.bias.data.fill_(0)
        init.xavier_uniform_(self.q_linear.weight, gain=1) # initialize linear layer
        self.v_linear.bias.data.fill_(0)
        init.xavier_uniform_(self.v_linear.weight, gain=1) # initialize linear layer
        self.k_linear.bias.data.fill_(0)
        init.xavier_uniform_(self.k_linear.weight, gain=1) # initialize linear layer
        self.fc.bias.data.fill_(0)
        init.xavier_uniform_(self.fc.weight, gain=1) # initialize linear layer



    def forward(self, x, mask):
        bs, l_x, _  = x.size()
        # x = x.transpose(1,2)
        k = self.k_linear(x).view(bs, l_x, self.n_head, self.d_k)
        q = self.q_linear(x).view(bs, l_x, self.n_head, self.d_k)
        v = self.v_linear(x).view(bs, l_x, self.n_head, self.d_k)
        q = q.permute(2, 0, 1, 3).contiguous().view(bs*self.n_head, l_x, self.d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(bs*self.n_head, l_x, self.d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(bs*self.n_head, l_x, self.d_k)
        mask = mask.unsqueeze(1).expand(-1, l_x, -1).repeat(self.n_head, 1, 1)
        
        attn = torch.bmm(q, k.transpose(1, 2)) * self.a
        attn = mask_logits(attn, mask)
        attn = F.softmax(attn, dim=2)
        attn = self.dropout(attn)
            
        out = torch.bmm(attn, v)
        out = out.view(self.n_head, bs, l_x, self.d_k).permute(1,2,0,3).contiguous().view(bs, l_x, self.d_model)
        out = self.fc(out)
        out = self.dropout(out)
        return out

class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        Wo = torch.empty(d_model, d_k * n_head)
        Wqs = [torch.empty(d_model, d_k) for _ in range(n_head)]
        Wks = [torch.empty(d_model, d_k) for _ in range(n_head)]
        Wvs = [torch.empty(d_model, d_k) for _ in range(n_head)]
        nn.init.kaiming_uniform_(Wo)
        for i in range(n_head):
            nn.init.xavier_uniform_(Wqs[i])
            nn.init.xavier_uniform_(Wks[i])
            nn.init.xavier_uniform_(Wvs[i])
        self.Wo = nn.Parameter(Wo)
        self.Wqs = nn.ParameterList([nn.Parameter(X) for X in Wqs])
        self.Wks = nn.ParameterList([nn.Parameter(X) for X in Wks])
        self.Wvs = nn.ParameterList([nn.Parameter(X) for X in Wvs])

    def forward(self, x, mask):
        WQs, WKs, WVs = [], [], []
        sqrt_d_k_inv = 1 / math.sqrt(d_k)
        x = x.transpose(1, 2)
        hmask = mask.unsqueeze(1)
        vmask = mask.unsqueeze(2)
        for i in range(n_head):
            WQs.append(torch.matmul(x, self.Wqs[i]))
            WKs.append(torch.matmul(x, self.Wks[i]))
            WVs.append(torch.matmul(x, self.Wvs[i]))
        heads = []
        for i in range(n_head):
            out = torch.bmm(WQs[i], WKs[i].transpose(1, 2))
            out = torch.mul(out, sqrt_d_k_inv)
            # not sure... I think `dim` should be 2 since it weighted each column of `WVs[i]`
            out = mask_logits(out, hmask)
            out = F.softmax(out, dim=2) * vmask
            headi = torch.bmm(out, WVs[i])
            heads.append(headi)
        head = torch.cat(heads, dim=2)
        out = torch.matmul(head, self.Wo)
        return out.transpose(1, 2)

def seq_and_vec(seq_len, vec):
    return vec.unsqueeze(1).repeat(1,seq_len,1)
def mask_logits(target, mask):
    return target * (1-mask) + mask * (-1e30)

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))