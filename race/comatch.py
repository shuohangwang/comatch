'''
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.
Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.
This software is provided by the copyright holder and creator "as is" and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

def masked_softmax(vector, seq_lens):
    mask = vector.new(vector.size()).zero_()
    for i in range(seq_lens.size(0)):
        mask[i,:,:seq_lens[i]] = 1
    mask = Variable(mask, requires_grad=False)

    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=-1)
    else:
        result = torch.nn.functional.softmax(vector * mask, dim=-1)
        result = result * mask
        result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result

class MatchNet(nn.Module):
    def __init__(self, mem_dim, dropoutP):
        super(MatchNet, self).__init__()
        self.map_linear = nn.Linear(2*mem_dim, 2*mem_dim)
        self.trans_linear = nn.Linear(mem_dim, mem_dim)
        self.drop_module = nn.Dropout(dropoutP)

    def forward(self, inputs):
        proj_p, proj_q, seq_len = inputs
        trans_q = self.trans_linear(proj_q)
        att_weights = proj_p.bmm( torch.transpose(proj_q, 1, 2) )
        att_norm = masked_softmax(att_weights, seq_len)

        att_vec = att_norm.bmm(proj_q)
        elem_min = att_vec - proj_p
        elem_mul = att_vec * proj_p
        all_con = torch.cat([elem_min,elem_mul], 2)
        output = nn.ReLU()(self.map_linear(all_con))
        return output

class MaskLSTM(nn.Module):
    def __init__(self, in_dim, out_dim, layers=1, batch_first=True, bidirectional=True, dropoutP = 0.3):
        super(MaskLSTM, self).__init__()
        self.lstm_module = nn.LSTM(in_dim, out_dim, layers, batch_first=batch_first, bidirectional=bidirectional, dropout=dropoutP)
        self.drop_module = nn.Dropout(dropoutP)

    def forward(self, inputs):
        input, seq_lens = inputs
        mask_in = input.new(input.size()).zero_()
        for i in range(seq_lens.size(0)):
            mask_in[i,:seq_lens[i]] = 1
        mask_in = Variable(mask_in, requires_grad=False)

        input_drop = self.drop_module(input*mask_in)

        H, _ = self.lstm_module(input_drop)

        mask = H.new(H.size()).zero_()
        for i in range(seq_lens.size(0)):
            mask[i,:seq_lens[i]] = 1
        mask = Variable(mask, requires_grad=False)

        output = H * mask

        return output

class CoMatch(nn.Module):
    def __init__(self, corpus, args):
        super(CoMatch, self).__init__()
        self.emb_dim = 300
        self.mem_dim = args.mem_dim
        self.dropoutP = args.dropoutP
        self.cuda_bool = args.cuda

        self.embs = nn.Embedding(len(corpus.dictionary), self.emb_dim)
        self.embs.weight.data.copy_(corpus.dictionary.embs)
        self.embs.weight.requires_grad = False

        self.encoder = MaskLSTM(self.emb_dim, self.mem_dim, dropoutP=self.dropoutP)
        self.l_encoder = MaskLSTM(self.mem_dim*8, self.mem_dim, dropoutP=self.dropoutP)
        self.h_encoder = MaskLSTM(self.mem_dim*2, self.mem_dim, dropoutP=0)

        self.match_module = MatchNet(self.mem_dim*2, self.dropoutP)
        self.rank_module = nn.Linear(self.mem_dim*2, 1)

        self.drop_module = nn.Dropout(self.dropoutP)

    def forward(self, inputs):
        documents, questions, options, _ = inputs
        d_word, d_h_len, d_l_len = documents
        o_word, o_h_len, o_l_len = options
        q_word, q_len = questions

        if self.cuda_bool: d_word, d_h_len, d_l_len, o_word, o_h_len, o_l_len, q_word, q_len = d_word.cuda(), d_h_len.cuda(), d_l_len.cuda(), o_word.cuda(), o_h_len.cuda(), o_l_len.cuda(), q_word.cuda(), q_len.cuda()
        d_embs = self.drop_module( Variable(self.embs(d_word), requires_grad=False) )
        o_embs = self.drop_module( Variable(self.embs(o_word), requires_grad=False) )
        q_embs = self.drop_module( Variable(self.embs(q_word), requires_grad=False) )

        d_hidden = self.encoder([d_embs.view(d_embs.size(0)*d_embs.size(1), d_embs.size(2), self.emb_dim), d_l_len.view(-1)] )
        o_hidden = self.encoder([o_embs.view(o_embs.size(0)*o_embs.size(1), o_embs.size(2), self.emb_dim), o_l_len.view(-1)])
        q_hidden = self.encoder([q_embs, q_len])

        d_hidden_3d = d_hidden.view(d_embs.size(0), d_embs.size(1) * d_embs.size(2), d_hidden.size(-1))
        d_hidden_3d_repeat = d_hidden_3d.repeat(1, o_embs.size(1), 1).view(d_hidden_3d.size(0)*o_embs.size(1), d_hidden_3d.size(1), d_hidden_3d.size(2))


        do_match = self.match_module([d_hidden_3d_repeat, o_hidden, o_l_len.view(-1)])
        dq_match = self.match_module([d_hidden_3d, q_hidden, q_len])

        dq_match_repeat = dq_match.repeat(1, o_embs.size(1), 1).view(dq_match.size(0)*o_embs.size(1), dq_match.size(1), dq_match.size(2))

        co_match= torch.cat([do_match, dq_match_repeat], -1)

        co_match_hier = co_match.view(d_embs.size(0)*o_embs.size(1)*d_embs.size(1), d_embs.size(2), -1)

        l_hidden = self.l_encoder([co_match_hier, d_l_len.repeat(1, o_embs.size(1)).view(-1)])
        l_hidden_pool, _ = l_hidden.max(1)

        h_hidden = self.h_encoder([l_hidden_pool.view(d_embs.size(0)*o_embs.size(1), d_embs.size(1), -1), d_h_len.view(-1, 1).repeat(1, o_embs.size(1)).view(-1)])
        h_hidden_pool, _ = h_hidden.max(1)

        o_rep = h_hidden_pool.view(d_embs.size(0), o_embs.size(1), -1)
        output = torch.nn.functional.log_softmax( self.rank_module(o_rep).squeeze(2) )

        return output
