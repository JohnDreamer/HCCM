# This file contains Att2in2, AdaAtt, AdaAttMO, TopDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# TopDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998
# However, it may not be identical to the author's architecture.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.utils as utils
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from .CaptionModel import CaptionModel


import copy
bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']
bad_endings += ['the']
import os


def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)

class HCModel16(CaptionModel):
    def __init__(self, opt, ix2glove=None):
        super(HCModel16, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = getattr(opt, 'max_length', 20) or opt.seq_length # maximum sample length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        self.use_bn = getattr(opt, 'use_bn', 0)

        self.ss_prob = 0.0 # Schedule sampling probability

        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                nn.ReLU(),
                                nn.Dropout(self.drop_prob_lm))
        self.fc_embed = nn.Sequential(nn.Linear(self.att_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
        # self.att_embed = nn.Sequential(*(
        #                             ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
        #                             (nn.Linear(self.att_feat_size, self.rnn_size),
        #                             nn.ReLU(),
        #                             nn.Dropout(self.drop_prob_lm))+
        #                             ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn==2 else ())))
        self.att_embed = nn.Sequential(nn.Linear(self.att_feat_size, self.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))

        # self.fc_embed1 = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
        #                               nn.ReLU(),
        #                               nn.Dropout(self.drop_prob_lm))
        #
        # self.att_embed1 = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
        #                                nn.ReLU(),
        #                                nn.Dropout(self.drop_prob_lm))

        # self.human_rela_embed = nn.Sequential(nn.Linear(self.att_feat_size, self.fc_feat_size),
        #                                 nn.ReLU(),
        #                                 nn.Dropout(self.drop_prob_lm))

        # self.fc_embed2 = nn.Sequential(nn.Linear(self.rnn_size, self.rnn_size),
        #                               nn.ReLU(),
        #                               nn.Dropout(self.drop_prob_lm))
        # self.att_embed2 = nn.Sequential(nn.Linear(self.rnn_size, self.rnn_size),
        #                               nn.ReLU(),
        #                               nn.Dropout(self.drop_prob_lm))

        # self.ho_embed = nn.Sequential(nn.Linear(self.rnn_size, self.rnn_size),
        #                                nn.ReLU(),
        #                                nn.Dropout(self.drop_prob_lm))

        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(*(reduce(lambda x,y:x+y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)

        # For remove bad endding
        self.vocab = opt.vocab
        self.bad_endings_ix = [int(k) for k,v in self.vocab.items() if v in bad_endings]

        # human centric
        self.core = TopDownCore15(opt)

        self.head = HakeModule2(opt)

        # self.hoh_fc = nn.Linear(self.rnn_size, self.rnn_size)

        # self.use_glove = opt.use_glove
        # if self.use_glove == 1:
        #     self.ix2glove = torch.zeros((len(ix2glove), 300))
        #     for ix, vec in ix2glove.items():
        #         self.ix2glove[ix] = torch.Tensor(vec)
        #     self.glove_fc = nn.Linear(3000, self.rnn_size)
        #     self.human_glove_merge_fc = nn.Linear(2*self.rnn_size, self.rnn_size)

        self.weight_thresh = 0.05

        # if os.path.isfile(self.memory_cell_path):
        #     print('load memory_cell from {0}'.format(self.memory_cell_path))
        #     memory_init = np.load(self.memory_cell_path)['memory_cell'][()]
        # else:
        print('create a new memory_cell')
        memory_init = np.random.rand(self.rnn_size, self.rnn_size) / 100
        memory_init = np.float32(memory_init)
        # self.memory_cell = torch.from_numpy(memory_init).cuda().requires_grad_()
        self.memory_cell = torch.from_numpy(memory_init).requires_grad_()


        #self.rela_mem = Memory_cell2(opt)
        # self.ssg_mem = Memory_cell2(opt)
        # self.img2sg = imge2sene_fc(opt)
        self.img2sg = nn.Sequential(nn.Linear(self.rnn_size, self.rnn_size),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    nn.Dropout(self.drop_prob_lm),
                                    nn.Linear(self.rnn_size, self.rnn_size))











    # def init_hidden_head(self, bsz):
    #     weight = next(self.parameters())
    #     return (weight.new_zeros(1, bsz, self.rnn_size),
    #             weight.new_zeros(1, bsz, self.rnn_size))

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        # return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
        #         weight.new_zeros(self.num_layers, bsz, self.rnn_size))
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def prepare_random_feature(self, att_feats):
        self.memory_cell = self.memory_cell.cuda(att_feats.get_device())
        att_feats_mem = self.img2sg(att_feats)
        # att_feats_mem = self.ssg_mem(att_feats_mem, self.memory_cell)
        att_shape = att_feats_mem.shape
        att_feats_mem = att_feats_mem.view(-1, att_shape[-1])
        att_feats_mem = torch.mm(att_feats_mem, self.memory_cell)
        att_feats_mem = att_feats_mem.view(att_shape)
        p_att_feats_mem = self.ctx2att(att_feats_mem)
        return att_feats_mem, p_att_feats_mem

    def _prepare_feature(self, human_rela_feats, other_feats, human_rela_masks, other_masks, body_feats, human_att, obj_att, body_masks, fc_feats, att_feats, att_masks):
        # tmp_masks = torch.sum(human_rela_masks, dim=1)
        # tmp_masks1 = tmp_masks.copy().detach()
        # tmp_masks1[tmp_masks==0] = 1
        # fc_feats = torch.sum(human_rela_feats, dim=1) * tmp_masks.unsqueeze(-1) / tmp_masks1.unsqueeze(-1)
        fc_feats_ori = self.fc_embed(fc_feats)
        att_feats_ori, att_masks_ori = self.clip_att(att_feats, att_masks)
        att_feats_ori = self.att_embed(att_feats_ori)
        p_att_feats_ori = self.ctx2att(att_feats_ori)


        # human_rela_feats = self.human_rela_embed(human_rela_feats)

        human_rela_feats = torch.cat([human_rela_feats,
                                      human_rela_feats.new_zeros(human_rela_feats.shape[0],
                                                                 3, human_rela_feats.shape[-1])], dim=1).contiguous()

        human_rela_masks = torch.cat([human_rela_masks,
                                      human_rela_masks.new_zeros(human_rela_masks.shape[0],
                                                                 3)], dim=1).contiguous()

        for b_i in range(human_rela_feats.shape[0]):
            h_index = int((torch.sum(human_rela_masks[b_i])).item())
            # human_rela_feats[b_i, h_index] = human_att[b_i]
            # human_rela_feats[b_i, h_index+1] = obj_att[b_i]
            # human_rela_masks[b_i, h_index] = 1
            # human_rela_masks[b_i, h_index + 1] = 1
            #
            # h_index += 2

            for h_i in range(3):
                if body_masks[b_i, h_i] == 1:
                    human_rela_feats[b_i, h_index] = body_feats[b_i, h_i]
                    # human_rela_feats[b_i, h_index + 1] = human_att[b_i, h_i]
                    # human_rela_feats[b_i, h_index + 2] = obj_att[b_i, h_i]
                    human_rela_masks[b_i, h_index] = 1
                    # human_rela_feats[b_i, h_index + 1] = 1
                    # human_rela_feats[b_i, h_index + 2] = 1
                    h_index += 1
                else:
                    break
        att_feats, att_masks = self.clip_att(human_rela_feats, human_rela_masks)
        fc_feats = torch.sum(att_feats, dim=1) / torch.sum(att_masks, dim=1).unsqueeze(-1)
        fc_feats = self.fc_embed(fc_feats)

        # if att_feats is not None:
        # att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        att_feats = self.att_embed(att_feats)

        # att_feats = torch.cat([att_feats,
        #                        att_feats.new_zeros(att_feats.shape[0],
        #                                         6, att_feats.shape[-1])], dim=1).contiguous()
        #
        # att_masks = torch.cat([att_masks,
        #                        att_masks.new_zeros(att_masks.shape[0],
        #                                         6)], dim=1).contiguous()

        # human_att = self.ho_embed(human_att)
        # obj_att = self.ho_embed(obj_att)
        # for b_i in range(att_feats.shape[0]):
        #     h_index = int((torch.sum(att_masks[b_i])).item())
        #     for h_i in range(3):
        #         if body_masks[b_i, h_i] == 1:
        #             att_feats[b_i, h_index] = human_att[b_i, h_i]
        #             att_feats[b_i, h_index+1] = obj_att[b_i, h_i]
        #             att_masks[b_i, h_index] = 1
        #             att_masks[b_i, h_index+1] = 1
        #
        #             h_index += 2
        #         else:
        #             break

        # att_feats, att_masks = self.clip_att(att_feats, att_masks)


        p_att_feats = self.ctx2att(att_feats)

        # tmp_masks = torch.sum(other_masks, dim=1)
        # tmp_masks1 = tmp_masks.clone().detach()
        # tmp_masks1[tmp_masks==0] = 1
        # fc_feats1 = torch.sum(other_feats, dim=1) * tmp_masks.unsqueeze(-1) / tmp_masks1.unsqueeze(-1)

        fc_feats1 = torch.sum(other_feats, dim=1) / (torch.sum(other_masks, dim=1).unsqueeze(-1) + 1e-10)
        fc_feats1 = self.fc_embed(fc_feats1)

        att_feats1, att_masks1 = self.clip_att(other_feats, other_masks)
        # att_feats1 = pack_wrapper(self.att_embed, att_feats1, att_masks1)
        att_feats1 = self.att_embed(att_feats1)

        p_att_feats1 = self.ctx2att(att_feats1)

        # fc_feats2 = torch.sum(human_feats, dim=1) / torch.sum(body_masks, dim=1).unsqueeze(-1)
        # fc_feats2 = self.fc_embed2(fc_feats2)
        #
        # att_feats2 = self.att_embed2(human_feats)
        # att_masks2 = body_masks
        #
        # p_att_feats2 = self.ctx2att(att_feats2)

        fc_feats = torch. cat([fc_feats_ori, fc_feats, fc_feats1], dim=-1)

        return fc_feats, att_feats, p_att_feats, att_masks, att_feats1, \
               p_att_feats1, att_masks1, att_feats_ori, p_att_feats_ori, att_masks_ori

    # def trunk_process(self, trunk, body_masks):
    #     trunk_fc = torch.sum(trunk, dim=1) / torch.sum(body_masks.view(trunk.shape[:2]), dim=1).unsqueeze(-1)
    #     trunk_fc = self.trunk_fc(trunk_fc)
    #     truck = self.trunk_att(trunk)
    #     p_trunk_att = self.ctx2att_trunk(trunk)
    #
        # return trunk_fc, trunk, p_trunk_att, body_masks

    def splitAttFeats(self, att_feats, att_masks, human_weight, obj_weight):

        human_weight = human_weight.view(-1, 3, human_weight.shape[-1])
        obj_weight = obj_weight.view(-1, 3, human_weight.shape[-1])
        human_masks = ((human_weight > self.weight_thresh) + (obj_weight > self.weight_thresh)) > 0
        tmp = torch.sum(human_weight + obj_weight, dim=1)
        tmp = ((tmp - tmp.max(dim=-1)[0].unsqueeze(-1)) == 0).float()
        human_masks = (torch.sum(human_masks, dim=1).float() + tmp) > 0

        att_feats = att_feats[:, :human_masks.shape[1]]
        att_masks = att_masks[:, :human_masks.shape[1]]

        human_att = torch.bmm(human_weight, att_feats)
        obj_att = torch.bmm(obj_weight, att_feats)


        human_masks_sum = torch.sum(human_masks, dim=1)
        max_len = torch.max(human_masks_sum)
        # print('xxxx', max_len)
        # if max_len == 0:
        #     human_masks = torch.sum(human_weight + obj_weight, dim=1).max(dim=-1)

        human_rela_feats = att_feats.new_zeros(att_feats.shape[0], max_len, att_feats.shape[-1])
        human_rela_masks = att_masks.new_zeros(att_masks.shape[0], max_len)
        for b_i in range(att_feats.shape[0]):
        #     # tmp = att_feats[b_i][human_masks[b_i]]
            human_rela_feats[b_i, :human_masks_sum[b_i]] = att_feats[b_i][human_masks[b_i]]
            human_rela_masks[b_i, :human_masks_sum[b_i]] = 1


        human_masks = 1 - human_masks
        human_masks = human_masks * att_masks.type_as(human_masks)
        human_masks_sum = torch.sum(human_masks, dim=1)
        max_len = torch.max(human_masks_sum)

        other_feats = att_feats.new_zeros(att_feats.shape[0], max_len, att_feats.shape[-1])
        other_masks = att_masks.new_zeros(att_masks.shape[0], max_len)
        for b_i in range(att_feats.shape[0]):
            other_feats[b_i, :human_masks_sum[b_i]] = att_feats[b_i][human_masks[b_i]]
            other_masks[b_i, :human_masks_sum[b_i]] = 1


        return human_rela_feats, other_feats, human_rela_masks, other_masks, human_att, obj_att






    def _forward(self, fc_feats, att_feats, seq, body_feats, part_feats, body_masks, part_masks, att_masks=None, hoi_train=False):




        # prepare the human feature
        # human_feats, p_human_feats, human_masks = self._prepare_body_part_feature(body_feats, part_feats, kp_feats, body_masks)
        # human_feats, p_human_feats, human_masks = None, None, None
        # body_feats, part_feats = self._prepare_body_part_feature(body_feats, part_feats)


        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost

        # state_head = self.init_hidden_head(batch_size)
        hoi, part_out, obj_out, human_feats, human_att, obj_att, human_weight, obj_weight = self.head(fc_feats, att_feats, body_feats,
                                                                               part_feats, att_masks, body_masks,
                                                                               part_masks)

        if hoi_train:
            return hoi, part_out, obj_out

        human_rela_feats, other_feats, human_rela_masks, other_masks, human_att, obj_att = self.splitAttFeats(att_feats, att_masks, human_weight, obj_weight)
        # other_feats, other_masks = att_feats, att_masks

        # if self.use_glove == 1:
        #     part_out = part_out.view(part_out.shape[0], 3, 10, part_out.shape[-1])
        #     _, part_state = torch.max(part_out.data, 3)
        #     self.ix2glove = self.ix2glove.to(fc_feats.device)
        #     part_state = self.ix2glove[part_state]
        #     part_state = part_state * part_masks.view(part_state.shape[0], 3, 10, 1)
        #     part_state = part_state.view(part_out.shape[0], 3, 3000)
        #     part_state = F.relu(self.glove_fc(part_state))

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        # Prepare the features
        fc_feats, att_feats, p_att_feats, att_masks, att_feats1, \
        p_att_feats1, att_masks1, att_feats_ori, p_att_feats_ori, att_masks_ori = \
            self._prepare_feature(human_rela_feats, other_feats, human_rela_masks, other_masks, body_feats, human_att, obj_att, body_masks, fc_feats, att_feats, att_masks)


        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size + 1)
        # trunk_fc, trunk, p_trunk_att, trunk_masks = self.trunk_process(trunks, body_masks)

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                    # prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                    prob_prev = torch.exp(outputs[:, i-1].detach()) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()          
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(it, fc_feats, att_feats, p_att_feats, att_masks, att_feats1,
                                                    p_att_feats1, att_masks1, att_feats_ori, p_att_feats_ori, att_masks_ori, human_feats, body_masks, state)
            outputs[:, i] = output

        return outputs

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, att_feats1, p_att_feats1, att_masks1, att_feats_ori, p_att_feats_ori, att_masks_ori, part_state, part_state_masks, state):
        # 'it' contains a word index
        xt = self.embed(it)

        att_feats_mem, p_att_feats_mem = self.prepare_random_feature(att_feats_ori)
        # output, state = self.core(xt, fc_feats, att_feats, p_att_feats, att_masks, human_feats, p_human_feats, human_masks, state)
        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, att_masks, att_feats1, p_att_feats1, att_masks1, att_feats_ori, p_att_feats_ori, att_masks_ori, att_feats_mem, p_att_feats_mem, part_state, part_state_masks, state)

        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state

    def _sample_beam(self, fc_feats, att_feats, body_feats, part_feats, body_masks, part_masks, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)



        hoi, part_out, obj_out, human_feats, human_att, obj_att, human_weight, obj_weight = self.head(fc_feats,
                                                                                                      att_feats,
                                                                                                      body_feats,
                                                                                                      part_feats,
                                                                                                      att_masks,
                                                                                                      body_masks,
                                                                                                      part_masks)
        human_rela_feats, other_feats, human_rela_masks, other_masks, human_att, obj_att = self.splitAttFeats(att_feats, att_masks,
                                                                                          human_weight, obj_weight)
        # other_feats, other_masks = att_feats, att_masks

        fc_feats, att_feats, p_att_feats, att_masks, att_feats1, \
        p_att_feats1, att_masks1, att_feats_ori, p_att_feats_ori, att_masks_ori = \
            self._prepare_feature(human_rela_feats, other_feats, human_rela_masks,
                                  other_masks, body_feats, human_att, obj_att, body_masks, fc_feats, att_feats, att_masks)



        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = fc_feats[k:k+1].expand(beam_size, fc_feats.size(1))
            tmp_att_feats = att_feats[k:k+1].expand(*((beam_size,)+att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = p_att_feats[k:k+1].expand(*((beam_size,)+p_att_feats.size()[1:])).contiguous()
            tmp_att_masks = att_masks[k:k+1].expand(*((beam_size,)+att_masks.size()[1:])).contiguous() if att_masks is not None else None

            tmp_att_feats1 = att_feats1[k:k + 1].expand(*((beam_size,) + att_feats1.size()[1:])).contiguous()
            tmp_p_att_feats1 = p_att_feats1[k:k + 1].expand(*((beam_size,) + p_att_feats1.size()[1:])).contiguous()
            tmp_att_masks1 = att_masks1[k:k + 1].expand(
                *((beam_size,) + att_masks1.size()[1:])).contiguous() if att_masks1 is not None else None

            tmp_att_feats_ori = att_feats_ori[k:k + 1].expand(*((beam_size,) + att_feats_ori.size()[1:])).contiguous()
            tmp_p_att_feats_ori = p_att_feats_ori[k:k + 1].expand(*((beam_size,) + p_att_feats_ori.size()[1:])).contiguous()
            tmp_att_masks_ori = att_masks_ori[k:k + 1].expand(
                *((beam_size,) + att_masks_ori.size()[1:])).contiguous() if att_masks_ori is not None else None

            tmp_part_state = human_feats[k:k + 1].expand(*((beam_size,) + human_feats.size()[1:])).contiguous()

            tmp_body_masks = body_masks[k:k + 1].expand(
                *((beam_size,) + body_masks.size()[1:])).contiguous() if body_masks is not None else None




            # tmp_trunk = trunk[k:k + 1].expand(*((beam_size,) + trunk.size()[1:])).contiguous()
            # tmp_p_trunk_att = p_trunk_att[k:k + 1].expand(*((beam_size,) + p_trunk_att.size()[1:])).contiguous()
            # tmp_trunk_masks = trunk_masks[k:k + 1].expand(*((beam_size,) + trunk_masks.size()[1:])).contiguous() if trunk_masks is not None else None
            # tmp_part_masks = part_masks[k:k + 1].expand(
            #     *((beam_size,) + part_masks.size()[1:])).contiguous() if part_masks is not None else None

            for t in range(1):
                if t == 0: # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, tmp_att_feats1,
                                                          tmp_p_att_feats1, tmp_att_masks1, tmp_att_feats_ori, tmp_p_att_feats_ori, tmp_att_masks_ori, tmp_part_state, tmp_body_masks, state)

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats,
                                                  tmp_att_masks, tmp_att_feats1, tmp_p_att_feats1,
                                                  tmp_att_masks1, tmp_att_feats_ori, tmp_p_att_feats_ori, tmp_att_masks_ori, tmp_part_state, tmp_body_masks, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, fc_feats, att_feats, body_feats, part_feats, body_masks, part_masks, att_masks=None, opt={}):

        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, body_feats, part_feats, body_masks, part_masks, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)



        hoi, part_out, obj_out, human_feats, human_att, obj_att, human_weight, obj_weight = self.head(fc_feats,
                                                                                                      att_feats,
                                                                                                      body_feats,
                                                                                                      part_feats,
                                                                                                      att_masks,
                                                                                                      body_masks,
                                                                                                      part_masks)


        human_rela_feats, other_feats, human_rela_masks, other_masks, human_att, obj_att = self.splitAttFeats(att_feats, att_masks,
                                                                                          human_weight, obj_weight)

        # other_feats, other_masks = att_feats, att_masks


        fc_feats, att_feats, p_att_feats, att_masks, att_feats1, \
        p_att_feats1, att_masks1, att_feats_ori, p_att_feats_ori, att_masks_ori = \
            self._prepare_feature(human_rela_feats, other_feats, human_rela_masks,
                                  other_masks, body_feats, human_att, obj_att, body_masks, fc_feats, att_feats, att_masks)



        trigrams = [] # will be a list of batch_size dictionaries

        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = att_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = att_feats.new_zeros(batch_size, dtype=torch.long)

            logprobs, state = self.get_logprobs_state(it, fc_feats, att_feats, p_att_feats, att_masks, att_feats1,
                                                    p_att_feats1, att_masks1, att_feats_ori, p_att_feats_ori, att_masks_ori, human_feats, body_masks, state)
            
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            if remove_bad_endings and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                prev_bad = np.isin(seq[:,t-1].data.cpu().numpy(), self.bad_endings_ix)
                # Impossible to generate remove_bad_endings
                tmp[torch.from_numpy(prev_bad.astype('uint8')), 0] = float('-inf')
                logprobs = logprobs + tmp

            # Mess with trigrams
            if block_trigrams and t >= 3:
                # Store trigram generated at last step
                prev_two_batch = seq[:,t-3:t-1]
                for i in range(batch_size): # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current  = seq[i][t-1]
                    if t == 3: # initialize
                        trigrams.append({prev_two: [current]}) # {LongTensor: list containing 1 int}
                    elif t > 3:
                        if prev_two in trigrams[i]: # add to list
                            trigrams[i][prev_two].append(current)
                        else: # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = seq[:,t-2:t]
                mask = torch.zeros(logprobs.size(), requires_grad=False).cuda() # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i,j] += 1
                # Apply mask to log probs
                #logprobs = logprobs - (mask * 1e9)
                alpha = 2.0 # = 4
                logprobs = logprobs + (mask * -0.693 * alpha) # ln(1/2) * alpha (alpha -> infty works best)

            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break
            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:,t] = it
            seqLogprobs[:,t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs





class TopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 3, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size) # h^1_t, \hat v
        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        lang_lstm_input = torch.cat([att, h_att], 1)
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state


class TopDownCore9(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore9, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.rnn_size = opt.rnn_size


        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size) # h^1_t, \hat v
        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, att_masks, att_feats1, p_att_feats1, att_masks1, state):
        prev_h = state[0][-1]

        fc_feats_ori = fc_feats[:, :self.rnn_size]
        fc_feats1 = fc_feats[:, self.rnn_size:2 * self.rnn_size]
        fc_feats2 = fc_feats[:, 2 * self.rnn_size:3 * self.rnn_size]

        att_lstm_input = torch.cat([prev_h, fc_feats1, xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        lang_lstm_input = torch.cat([att, h_att], 1)
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        att_lstm_input1 = torch.cat([prev_h, fc_feats2, xt], 1)
        h_att1, c_att1 = self.att_lstm(att_lstm_input1, (state[0][0], state[1][0]))
        att1 = self.attention(h_att1, att_feats1, p_att_feats1, att_masks1)
        lang_lstm_input1 = torch.cat([att1, h_lang], 1)
        h_lang1, c_lang1 = self.lang_lstm(lang_lstm_input1, (state[0][2], state[1][2]))


        output = F.dropout(h_lang1, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang, h_lang1]), torch.stack([c_att, c_lang, c_lang1]))

        return output, state


class TopDownCore10(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore10, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.rnn_size = opt.rnn_size


        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size) # h^1_t, \hat v
        self.attention = Attention(opt)
        self.attention1 = Attention(opt)


    def forward(self, xt, fc_feats, att_feats, p_att_feats, att_masks, att_feats1, p_att_feats1, att_masks1, state):
        prev_h = state[0][-1]

        fc_feats_ori = fc_feats[:, :self.rnn_size]
        fc_feats1 = fc_feats[:, self.rnn_size:2 * self.rnn_size]
        fc_feats2 = fc_feats[:, 2 * self.rnn_size:3 * self.rnn_size]

        att_lstm_input = torch.cat([prev_h, fc_feats_ori, xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        lang_lstm_input = torch.cat([att, h_att], 1)
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        # att_lstm_input1 = torch.cat([prev_h, fc_feats2, xt], 1)
        # h_att1, c_att1 = self.att_lstm(att_lstm_input1, (state[0][0], state[1][0]))
        att1 = self.attention1(h_lang, att_feats1, p_att_feats1, att_masks1)
        lang_lstm_input1 = torch.cat([att1, h_lang], 1)
        h_lang1, c_lang1 = self.lang_lstm(lang_lstm_input1, (state[0][2], state[1][2]))


        output = F.dropout(h_lang1, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang, h_lang1]), torch.stack([c_att, c_lang, c_lang1]))

        return output, state


class TopDownCore11(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore11, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.rnn_size = opt.rnn_size

        self.att_lstm0 = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 3, opt.rnn_size) # h^1_t, \hat v

        self.attention = Attention(opt)
        self.attention1 = Attention(opt)
        self.attention0 = Attention(opt)



    def forward(self, xt, fc_feats, att_feats, p_att_feats, att_masks, att_feats1, p_att_feats1, att_masks1, state):
        prev_h = state[0][-1]

        fc_feats_ori = fc_feats[:, :self.rnn_size]
        fc_feats1 = fc_feats[:, self.rnn_size:2 * self.rnn_size]
        fc_feats2 = fc_feats[:, 2 * self.rnn_size:3 * self.rnn_size]

        # fc_feats_ori = (fc_feats_ori + fc_feats1) / 2.0

        att_lstm_input0 = torch.cat([prev_h, fc_feats_ori, xt], 1)

        h_att0, c_att0 = self.att_lstm0(att_lstm_input0, (state[0][0], state[1][0]))
        att0 = self.attention0(h_att0, att_feats1, p_att_feats1, att_masks1)

        att_lstm_input = torch.cat([prev_h, att0, xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][1], state[1][1]))

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)
        att1 = self.attention1(h_att, att_feats1, p_att_feats1, att_masks1)

        lang_lstm_input = torch.cat([att, att1, h_att], 1)
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][2], state[1][2]))

        # att_lstm_input1 = torch.cat([prev_h, fc_feats2, xt], 1)
        # h_att1, c_att1 = self.att_lstm(att_lstm_input1, (state[0][0], state[1][0]))
        # att1 = self.attention1(h_lang, att_feats1, p_att_feats1, att_masks1)
        # lang_lstm_input1 = torch.cat([att1, h_lang], 1)
        # h_lang1, c_lang1 = self.lang_lstm(lang_lstm_input1, (state[0][2], state[1][2]))


        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att0, h_att, h_lang]), torch.stack([c_att0, c_att, c_lang]))

        return output, state


class TopDownCore12(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore12, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.rnn_size = opt.rnn_size

        # self.att_lstm0 = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size) # h^1_t, \hat v

        self.attention = Attention(opt)
        self.attention1 = Attention(opt)

        self.human_gate = nn.Sequential(nn.Linear(2*opt.rnn_size, opt.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(opt.drop_prob_lm))
        # self.attention0 = Attention(opt)



    def forward(self, xt, fc_feats, att_feats, p_att_feats, att_masks, att_feats1, p_att_feats1, att_masks1, state):
        prev_h = state[0][-1]

        fc_feats_ori = fc_feats[:, :self.rnn_size]
        fc_feats1 = fc_feats[:, self.rnn_size:2 * self.rnn_size]
        fc_feats2 = fc_feats[:, 2 * self.rnn_size:3 * self.rnn_size]

        # fc_feats_ori = (fc_feats_ori + fc_feats1) / 2.0

        # att_lstm_input0 = torch.cat([prev_h, fc_feats_ori, xt], 1)

        # h_att0, c_att0 = self.att_lstm0(att_lstm_input0, (state[0][0], state[1][0]))
        # att0 = self.attention0(h_att0, att_feats1, p_att_feats1, att_masks1)

        att_lstm_input = torch.cat([prev_h, fc_feats_ori, xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        gate = F.sigmoid(self.human_gate(torch.cat([fc_feats1, h_att], dim=-1)))

        att1 = self.attention1(h_att, att_feats1, p_att_feats1, att_masks1)
        att1 = att * gate + att1

        lang_lstm_input = torch.cat([att1, h_att], 1)
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        # att_lstm_input1 = torch.cat([prev_h, fc_feats2, xt], 1)
        # h_att1, c_att1 = self.att_lstm(att_lstm_input1, (state[0][0], state[1][0]))
        # att1 = self.attention1(h_lang, att_feats1, p_att_feats1, att_masks1)
        # lang_lstm_input1 = torch.cat([att1, h_lang], 1)
        # h_lang1, c_lang1 = self.lang_lstm(lang_lstm_input1, (state[0][2], state[1][2]))


        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state

class TopDownCore13(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore13, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.rnn_size = opt.rnn_size

        # self.att_lstm0 = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size) # h^1_t, \hat v

        self.attention = Attention(opt)
        self.attention_obj = Attention(opt)

        self.attention_ori = Attention(opt)

        self.human_gate = nn.Sequential(nn.Linear(2*opt.rnn_size, opt.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(opt.drop_prob_lm))
        self.obj_gate = nn.Sequential(nn.Linear(2 * opt.rnn_size, opt.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(opt.drop_prob_lm))
        # self.attention0 = Attention(opt)



    def forward(self, xt, fc_feats, att_feats, p_att_feats, att_masks, att_feats1, p_att_feats1, att_masks1, att_feats_ori, p_att_feats_ori, att_masks_ori, state):
        prev_h = state[0][-1]

        fc_feats_ori = fc_feats[:, :self.rnn_size]
        fc_feats1 = fc_feats[:, self.rnn_size:2 * self.rnn_size]
        fc_feats2 = fc_feats[:, 2 * self.rnn_size:3 * self.rnn_size]

        # fc_feats_ori = (fc_feats_ori + fc_feats1) / 2.0

        # att_lstm_input0 = torch.cat([prev_h, fc_feats_ori, xt], 1)

        # h_att0, c_att0 = self.att_lstm0(att_lstm_input0, (state[0][0], state[1][0]))
        # att0 = self.attention0(h_att0, att_feats1, p_att_feats1, att_masks1)

        att_lstm_input = torch.cat([prev_h, fc_feats_ori, xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)
        gate = torch.sigmoid(self.human_gate(torch.cat([fc_feats1, h_att], dim=-1)))

        att_obj = self.attention_obj(h_att, att_feats1, p_att_feats1, att_masks1)
        gate_obj = torch.sigmoid(self.obj_gate(torch.cat([fc_feats2, h_att], dim=-1)))

        att_ori = self.attention_ori(h_att, att_feats_ori, p_att_feats_ori, att_masks_ori)
        att_ori = att * gate + att_obj * gate_obj + att_ori

        lang_lstm_input = torch.cat([att_ori, h_att], 1)
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        # att_lstm_input1 = torch.cat([prev_h, fc_feats2, xt], 1)
        # h_att1, c_att1 = self.att_lstm(att_lstm_input1, (state[0][0], state[1][0]))
        # att1 = self.attention1(h_lang, att_feats1, p_att_feats1, att_masks1)
        # lang_lstm_input1 = torch.cat([att1, h_lang], 1)
        # h_lang1, c_lang1 = self.lang_lstm(lang_lstm_input1, (state[0][2], state[1][2]))


        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state


class TopDownCore14(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore14, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.rnn_size = opt.rnn_size

        # self.att_lstm0 = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 3, opt.rnn_size) # h^1_t, \hat v

        self.attention = Attention(opt)
        self.attention_obj = Attention(opt)

        self.attention_ori = Attention(opt)
        self.attention_mem = Attention(opt)


        self.human_gate = nn.Sequential(nn.Linear(2*opt.rnn_size, opt.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(opt.drop_prob_lm))
        self.obj_gate = nn.Sequential(nn.Linear(2 * opt.rnn_size, opt.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(opt.drop_prob_lm))
        # self.attention0 = Attention(opt)



    def forward(self, xt, fc_feats, att_feats, p_att_feats, att_masks, att_feats1, p_att_feats1, att_masks1, att_feats_ori, p_att_feats_ori, att_masks_ori, att_feats_mem, p_att_feats_mem, state):
        prev_h = state[0][-1]

        fc_feats_ori = fc_feats[:, :self.rnn_size]
        fc_feats1 = fc_feats[:, self.rnn_size:2 * self.rnn_size]
        fc_feats2 = fc_feats[:, 2 * self.rnn_size:3 * self.rnn_size]

        # fc_feats_ori = (fc_feats_ori + fc_feats1) / 2.0

        # att_lstm_input0 = torch.cat([prev_h, fc_feats_ori, xt], 1)

        # h_att0, c_att0 = self.att_lstm0(att_lstm_input0, (state[0][0], state[1][0]))
        # att0 = self.attention0(h_att0, att_feats1, p_att_feats1, att_masks1)

        att_lstm_input = torch.cat([prev_h, fc_feats_ori, xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)
        gate = torch.sigmoid(self.human_gate(torch.cat([fc_feats1, h_att], dim=-1)))

        att_obj = self.attention_obj(h_att, att_feats1, p_att_feats1, att_masks1)
        gate_obj = torch.sigmoid(self.obj_gate(torch.cat([fc_feats2, h_att], dim=-1)))

        att_ori = self.attention_ori(h_att, att_feats_ori, p_att_feats_ori, att_masks_ori)
        att_ori = att * gate + att_obj * gate_obj + att_ori

        att_mem = self.attention_mem(h_att, att_feats_mem, p_att_feats_mem, att_masks_ori)

        lang_lstm_input = torch.cat([att_mem, att_ori, h_att], 1)
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        # att_lstm_input1 = torch.cat([prev_h, fc_feats2, xt], 1)
        # h_att1, c_att1 = self.att_lstm(att_lstm_input1, (state[0][0], state[1][0]))
        # att1 = self.attention1(h_lang, att_feats1, p_att_feats1, att_masks1)
        # lang_lstm_input1 = torch.cat([att1, h_lang], 1)
        # h_lang1, c_lang1 = self.lang_lstm(lang_lstm_input1, (state[0][2], state[1][2]))


        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state


class TopDownCore15(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore15, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.rnn_size = opt.rnn_size

        # self.att_lstm0 = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 3, opt.rnn_size) # h^1_t, \hat v

        self.attention = Attention(opt)
        self.attention_obj = Attention(opt)

        self.attention_ori = Attention(opt)
        self.attention_mem = Attention(opt)




        self.human_gate = nn.Sequential(nn.Linear(2*opt.rnn_size, opt.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(opt.drop_prob_lm))
        self.obj_gate = nn.Sequential(nn.Linear(2 * opt.rnn_size, opt.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(opt.drop_prob_lm))
        # self.attention0 = Attention(opt)

        # self.attention_part = Attention(opt)
        # self.part_gate = nn.Sequential(nn.Linear(2 * opt.rnn_size, opt.rnn_size),
        #                               nn.ReLU(),
        #                               nn.Dropout(opt.drop_prob_lm))
        # self.part_ctt2att = nn.Linear(opt.rnn_size, opt.att_hid_size)
        self.part_cali = nn.Sequential(
            nn.Linear(opt.rnn_size + opt.rnn_size, opt.rnn_size),
            nn.ReLU(),
            nn.Dropout(opt.drop_prob_lm),
            nn.Linear(opt.rnn_size, opt.rnn_size)
        )



    def forward(self, xt, fc_feats, att_feats, p_att_feats, att_masks, att_feats1, p_att_feats1, att_masks1, att_feats_ori, p_att_feats_ori, att_masks_ori, att_feats_mem, p_att_feats_mem, part_state, part_state_masks, state):
        prev_h = state[0][-1]

        fc_feats_ori = fc_feats[:, :self.rnn_size]
        fc_feats1 = fc_feats[:, self.rnn_size:2 * self.rnn_size]
        fc_feats2 = fc_feats[:, 2 * self.rnn_size:3 * self.rnn_size]

        # fc_feats_ori = (fc_feats_ori + fc_feats1) / 2.0

        # att_lstm_input0 = torch.cat([prev_h, fc_feats_ori, xt], 1)

        # h_att0, c_att0 = self.att_lstm0(att_lstm_input0, (state[0][0], state[1][0]))
        # att0 = self.attention0(h_att0, att_feats1, p_att_feats1, att_masks1)

        att_lstm_input = torch.cat([prev_h, fc_feats_ori, xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)
        gate = torch.sigmoid(self.human_gate(torch.cat([fc_feats1, h_att], dim=-1)))

        # att_part = self.attention_part(att, part_state, self.part_ctt2att(part_state), part_state_masks)
        # fc_part = torch.sum(part_state, dim=1) / torch.sum(part_state_masks, dim=1).unsqueeze(-1)
        # gate_part = torch.sigmoid(self.part_gate(torch.cat([att, fc_part], dim=-1)))
        att_part = torch.sum(part_state, dim=1) / (torch.sum(part_state_masks, dim=1).unsqueeze(-1)+1e-10)
        att_part = self.part_cali(torch.cat([att_part, att], dim=-1))

        att = att + att_part

        att_obj = self.attention_obj(h_att, att_feats1, p_att_feats1, att_masks1)
        gate_obj = torch.sigmoid(self.obj_gate(torch.cat([fc_feats2, h_att], dim=-1)))

        att_ori = self.attention_ori(h_att, att_feats_ori, p_att_feats_ori, att_masks_ori)
        att_ori = att * gate + att_obj * gate_obj + att_ori

        att_mem = self.attention_mem(h_att, att_feats_mem, p_att_feats_mem, att_masks_ori)

        lang_lstm_input = torch.cat([att_mem, att_ori, h_att], 1)
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        # att_lstm_input1 = torch.cat([prev_h, fc_feats2, xt], 1)
        # h_att1, c_att1 = self.att_lstm(att_lstm_input1, (state[0][0], state[1][0]))
        # att1 = self.attention1(h_lang, att_feats1, p_att_feats1, att_masks1)
        # lang_lstm_input1 = torch.cat([att1, h_lang], 1)
        # h_lang1, c_lang1 = self.lang_lstm(lang_lstm_input1, (state[0][2], state[1][2]))


        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state


############################################################################
# Notice:
# StackAtt and DenseAtt are models that I randomly designed.
# They are not related to any paper.
############################################################################


class Attention(nn.Module):
    def __init__(self, opt, input_size=1):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(input_size*self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = torch.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        
        weight = F.softmax(dot, dim=1)                             # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / (weight.sum(1, keepdim=True)+1e-10) # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1)) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return att_res


class Attention5(nn.Module):
    def __init__(self, opt, input_size=1):
        super(Attention5, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(input_size * self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)

        att_h = self.h2att(h)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = torch.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size

        weight = F.softmax(dot, dim=1)  # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / (weight.sum(1, keepdim=True)+1e-10)  # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1))  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size
        p_att_res = torch.bmm(weight.unsqueeze(1), att).squeeze(1)  # batch * att_feat_size

        return att_res, p_att_res


class Attention4(nn.Module):
    def __init__(self, opt, input_size):
        super(Attention4, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(input_size*self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)

        att_h = self.h2att(h)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = torch.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size

        weight = F.softmax(dot, dim=1)  # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / (1e-10+weight.sum(1, keepdim=True))  # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1))  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size

        return att_res


class Attention1(nn.Module):
    def __init__(self, opt):
        super(Attention1, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)

        att_h = self.h2att(h)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = torch.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size

        weight = F.gumbel_softmax(dot)  # batch * att_size

        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1))  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size

        return att_res


class Attention2(nn.Module):
    def __init__(self, opt, input_size=1):
        super(Attention2, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(input_size*self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)

        att_h = self.h2att(h)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = torch.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size

        weight = F.softmax(dot, dim=1)  # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / (weight.sum(1, keepdim=True)+1e-10)  # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1))  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size

        return att_res, weight


class Memory_cell2(nn.Module):
    def __init__(self, opt):
        """
        a_i = h^T*m_i
        a_i: 1*1

        h: R*1
        m_i: R*1
        M=[m_1,m_2,...,m_K]^T: K*R
        att = softmax(a): K*1
        h_out = M*att: N*R

        :param opt:
        """
        super(Memory_cell2, self).__init__()
        self.R = opt.rnn_size
        self.V = opt.att_hid_size

        self.W = nn.Linear(self.V, 1)

    def forward(self, h, M):
        M_size = M.size()  # K*R
        h_size = h.size()  # N*T*R
        h = h.view(-1, h_size[2]) # (N*T)*R
        att = torch.mm(h, torch.t(M)) #(N*T)*K
        att = F.softmax(att, dim=1) #(N*T)*K
        #att_sum = torch.sum(att, dim=1)
        att_max = torch.max(att,dim=1)
        max_index = torch.argmax(att,dim=1)
        att_res = torch.mm(att, M)  #(N*T)*K * K*R->(N*T)*R
        att_res = att_res.view([h_size[0], h_size[1], h_size[2]])
        return att_res


class HakeModule(nn.Module):
    def __init__(self, opt):
        super(HakeModule, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.body_part_to_lantern = nn.Sequential(
            nn.Linear(3*self.rnn_size, 2048),
            nn.ReLU(),
            nn.Dropout(opt.drop_prob_lm),
            nn.Linear(2048, self.rnn_size),
            nn.ReLU(),
            nn.Dropout(opt.drop_prob_lm)
        )


        self.logit_part = nn.Sequential(
            nn.Linear(self.rnn_size, 2048),
            nn.ReLU(),
            nn.Dropout(opt.drop_prob_lm),
            nn.Linear(2048, 62)
        )

        self.logit_hoi = nn.Sequential(
            nn.Linear(2*self.rnn_size, 2048),
            nn.ReLU(),
            nn.Dropout(opt.drop_prob_lm),
            nn.Linear(2048, 117)
        )

        self.logit_obj = nn.Sequential(
            nn.Linear(self.rnn_size, 2048),
            nn.ReLU(),
            nn.Dropout(opt.drop_prob_lm),
            nn.Linear(2048, 81)
        )


        self.ctx2att = nn.Linear(opt.rnn_size, opt.att_hid_size)
        self.p_body_fc = nn.Linear(opt.rnn_size, opt.rnn_size)


        self.human_attention = Attention2(opt)
        self.obj_attention = Attention2(opt)

        self.fc_embed = nn.Sequential(nn.Linear(opt.fc_feat_size, opt.rnn_size),
                                      nn.ReLU(),
                                      nn.Dropout(opt.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(opt.att_feat_size),) if opt.use_bn else ()) +
                (nn.Linear(opt.att_feat_size, opt.rnn_size),
                 nn.ReLU(),
                 nn.Dropout(opt.drop_prob_lm)) +
                ((nn.BatchNorm1d(opt.rnn_size),) if opt.use_bn == 2 else ())))

        self.body_embed = nn.Sequential(nn.Linear(opt.fc_feat_size, opt.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(opt.drop_prob_lm))
        self.part_embed = nn.Sequential(nn.Linear(opt.fc_feat_size, opt.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(opt.drop_prob_lm))

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        fc_feats = self.fc_embed(fc_feats)
        # if att_feats is not None:
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation comsumptions.
        # p_att_feats = self.ctx2att(att_feats)
        # else:
        # p_att_feats = None

        return fc_feats, att_feats, att_masks

    def _prepare_body_part_feature(self, body_feat, part_feat):
        body_feat = self.body_embed(body_feat)
        part_feat = self.part_embed(part_feat)
        # hake_fc_feat = self.hake_fc_embed(hake_fc_feat)
        return body_feat, part_feat

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks



    def forward(self, fc_feats, att_feats, body_feats, part_feats, att_masks, body_masks, part_masks):

        fc_feats, att_feats, att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        body_feats, part_feats = self._prepare_body_part_feature(body_feats, part_feats)


        p_fc_feats = fc_feats.unsqueeze(1).expand_as(body_feats)
        p_body_feats = F.relu(self.p_body_fc(body_feats))
        p_body_feats = torch.cat([p_body_feats, body_feats, p_fc_feats], dim=-1)

        part_feats = part_feats.view(part_feats.shape[0], body_feats.shape[1], -1, part_feats.shape[-1])
        pp_fc_feats = fc_feats.view(fc_feats.shape[0],1,1,fc_feats.shape[1]).expand_as(part_feats)
        part_feats = torch.cat([part_feats, body_feats.unsqueeze(2).expand_as(part_feats), pp_fc_feats], dim=-1)

        body_feats = self.body_part_to_lantern(p_body_feats)
        part_feats = self.body_part_to_lantern(part_feats)

        part_feats = part_feats * part_masks.view(part_feats.shape[:3]).unsqueeze(-1)

        human_feats = torch.cat([body_feats.unsqueeze(2), part_feats], dim=2)
        human_feats = torch.sum(human_feats, dim=2) / (torch.sum(part_masks.view(part_feats.shape[:3]), dim=2)+1).unsqueeze(-1)

        p_att_feats = self.ctx2att(att_feats)
        ex_att_feats = att_feats.unsqueeze(1).expand(-1, human_feats.shape[1], -1, -1)
        ex_att_feats = ex_att_feats.contiguous().view(-1, ex_att_feats.shape[-2], ex_att_feats.shape[-1])
        ex_p_att_feats = p_att_feats.unsqueeze(1).expand(-1, human_feats.shape[1], -1, -1)
        ex_p_att_feats = ex_p_att_feats.contiguous().view(-1, ex_p_att_feats.shape[-2], ex_p_att_feats.shape[-1])
        ex_att_masks = att_masks.unsqueeze(1).expand(-1, human_feats.shape[1], -1)
        ex_att_masks = ex_att_masks.contiguous().view(-1, ex_att_masks.shape[-1])

        h_att = human_feats.view(-1, human_feats.shape[-1])
        obj_att, obj_weight = self.obj_attention(h_att, ex_att_feats, ex_p_att_feats, ex_att_masks)
        obj_att = obj_att.view(human_feats.shape)

        human_att, human_weight = self.human_attention(h_att, ex_att_feats, ex_p_att_feats, ex_att_masks)
        human_att = human_att.view(human_feats.shape)

        hoi_feats = torch.cat([human_att, obj_att.detach()], dim=-1)
        # hoi_feats = human_att

        part_out = self.logit_part(part_feats)
        hoi_out = self.logit_hoi(hoi_feats)
        obj_out = self.logit_obj(obj_att)

        part_out = part_out.view(part_out.shape[0], -1, part_out.shape[-1])
        human_weight = human_weight.view(body_masks.shape[0], body_masks.shape[1], -1)
        obj_weight = obj_weight.view(body_masks.shape[0], body_masks.shape[1], -1)
        human_weight = human_weight*body_masks.unsqueeze(-1)
        obj_weight = obj_weight*body_masks.unsqueeze(-1)



        return hoi_out, part_out, obj_out, human_feats, human_att, obj_att, human_weight, obj_weight


class HakeModule1(nn.Module):
    def __init__(self, opt):
        super(HakeModule1, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.body_part_to_lantern = nn.Sequential(
            nn.Linear(3*self.rnn_size, 2048),
            nn.ReLU(),
            nn.Dropout(opt.drop_prob_lm),
            nn.Linear(2048, self.rnn_size),
            nn.ReLU(),
            nn.Dropout(opt.drop_prob_lm)
        )


        self.logit_part = nn.Sequential(
            nn.Linear(self.rnn_size, 2048),
            nn.ReLU(),
            nn.Dropout(opt.drop_prob_lm),
            nn.Linear(2048, 62)
        )

        self.logit_hoi = nn.Sequential(
            nn.Linear(1*self.rnn_size, 2048),
            nn.ReLU(),
            nn.Dropout(opt.drop_prob_lm),
            nn.Linear(2048, 117)
        )

        self.logit_obj = nn.Sequential(
            nn.Linear(self.rnn_size, 2048),
            nn.ReLU(),
            nn.Dropout(opt.drop_prob_lm),
            nn.Linear(2048, 81)
        )


        self.ctx2att = nn.Linear(opt.rnn_size, opt.att_hid_size)
        self.p_body_fc = nn.Linear(opt.rnn_size, opt.rnn_size)
        # self.obj_att_fc = nn.Linear(opt.rnn_size, opt.rnn_size)


        self.human_attention = Attention2(opt,2)
        self.obj_attention = Attention2(opt)

        self.fc_embed = nn.Sequential(nn.Linear(opt.fc_feat_size, opt.rnn_size),
                                      nn.ReLU(),
                                      nn.Dropout(opt.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(opt.att_feat_size),) if opt.use_bn else ()) +
                (nn.Linear(opt.att_feat_size, opt.rnn_size),
                 nn.ReLU(),
                 nn.Dropout(opt.drop_prob_lm)) +
                ((nn.BatchNorm1d(opt.rnn_size),) if opt.use_bn == 2 else ())))

        self.body_embed = nn.Sequential(nn.Linear(opt.fc_feat_size, opt.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(opt.drop_prob_lm))
        self.part_embed = nn.Sequential(nn.Linear(opt.fc_feat_size, opt.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(opt.drop_prob_lm))

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        fc_feats = self.fc_embed(fc_feats)
        # if att_feats is not None:
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation comsumptions.
        # p_att_feats = self.ctx2att(att_feats)
        # else:
        # p_att_feats = None

        return fc_feats, att_feats, att_masks

    def _prepare_body_part_feature(self, body_feat, part_feat):
        body_feat = self.body_embed(body_feat)
        part_feat = self.part_embed(part_feat)
        # hake_fc_feat = self.hake_fc_embed(hake_fc_feat)
        return body_feat, part_feat

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks



    def forward(self, fc_feats, att_feats, body_feats, part_feats, att_masks, body_masks, part_masks):

        fc_feats, att_feats, att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        body_feats, part_feats = self._prepare_body_part_feature(body_feats, part_feats)


        p_fc_feats = fc_feats.unsqueeze(1).expand_as(body_feats)
        p_body_feats = F.relu(self.p_body_fc(body_feats))
        p_body_feats = torch.cat([p_body_feats, body_feats, p_fc_feats], dim=-1)

        part_feats = part_feats.view(part_feats.shape[0], body_feats.shape[1], -1, part_feats.shape[-1])
        pp_fc_feats = fc_feats.view(fc_feats.shape[0],1,1,fc_feats.shape[1]).expand_as(part_feats)
        part_feats = torch.cat([part_feats, body_feats.unsqueeze(2).expand_as(part_feats), pp_fc_feats], dim=-1)

        body_feats = self.body_part_to_lantern(p_body_feats)
        part_feats = self.body_part_to_lantern(part_feats)

        part_feats = part_feats * part_masks.view(part_feats.shape[:3]).unsqueeze(-1)

        human_feats = torch.cat([body_feats.unsqueeze(2), part_feats], dim=2)
        human_feats = torch.sum(human_feats, dim=2) / (torch.sum(part_masks.view(part_feats.shape[:3]), dim=2)+1).unsqueeze(-1)

        p_att_feats = self.ctx2att(att_feats)
        ex_att_feats = att_feats.unsqueeze(1).expand(-1, human_feats.shape[1], -1, -1)
        ex_att_feats = ex_att_feats.contiguous().view(-1, ex_att_feats.shape[-2], ex_att_feats.shape[-1])
        ex_p_att_feats = p_att_feats.unsqueeze(1).expand(-1, human_feats.shape[1], -1, -1)
        ex_p_att_feats = ex_p_att_feats.contiguous().view(-1, ex_p_att_feats.shape[-2], ex_p_att_feats.shape[-1])
        ex_att_masks = att_masks.unsqueeze(1).expand(-1, human_feats.shape[1], -1)
        ex_att_masks = ex_att_masks.contiguous().view(-1, ex_att_masks.shape[-1])

        h_att = human_feats.view(-1, human_feats.shape[-1])
        obj_att, obj_weight = self.obj_attention(h_att, ex_att_feats, ex_p_att_feats, ex_att_masks)
        obj_att = obj_att.view(human_feats.shape)

        # tmp_obj_att = F.relu(self.obj_att_fc())
        h_att = torch.cat([h_att, obj_att.detach().view(-1, human_feats.shape[-1])], dim=-1)

        human_att, human_weight = self.human_attention(h_att, ex_att_feats, ex_p_att_feats, ex_att_masks)
        human_att = human_att.view(human_feats.shape)

        # hoi_feats = torch.cat([human_att, obj_att.detach()], dim=-1)
        hoi_feats = human_att

        part_out = self.logit_part(part_feats)
        hoi_out = self.logit_hoi(hoi_feats)
        obj_out = self.logit_obj(obj_att)

        part_out = part_out.view(part_out.shape[0], -1, part_out.shape[-1])
        human_weight = human_weight.view(body_masks.shape[0], body_masks.shape[1], -1)
        obj_weight = obj_weight.view(body_masks.shape[0], body_masks.shape[1], -1)
        human_weight = human_weight*body_masks.unsqueeze(-1)
        obj_weight = obj_weight*body_masks.unsqueeze(-1)



        return hoi_out, part_out, obj_out, human_feats, human_att, obj_att, human_weight, obj_weight


class HakeModule2(nn.Module):
    def __init__(self, opt):
        super(HakeModule2, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.body_part_to_lantern = nn.Sequential(
            nn.Linear(3 * self.rnn_size, 2048),
            nn.ReLU(),
            nn.Dropout(opt.drop_prob_lm),
            nn.Linear(2048, self.rnn_size),
            nn.ReLU(),
            nn.Dropout(opt.drop_prob_lm)
        )

        self.logit_part = nn.Sequential(
            nn.Linear(self.rnn_size, 2048),
            nn.ReLU(),
            nn.Dropout(opt.drop_prob_lm),
            nn.Linear(2048, 62)
        )

        self.logit_hoi = nn.Sequential(
            nn.Linear(1 * self.rnn_size, 2048),
            nn.ReLU(),
            nn.Dropout(opt.drop_prob_lm),
            nn.Linear(2048, 117)
        )

        self.logit_obj = nn.Sequential(
            nn.Linear(self.rnn_size, 2048),
            nn.ReLU(),
            nn.Dropout(opt.drop_prob_lm),
            nn.Linear(2048, 81)
        )

        self.ctx2att = nn.Linear(opt.rnn_size, opt.att_hid_size)
        self.p_body_fc = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.obj_att_fc = nn.Linear(opt.rnn_size, opt.rnn_size)

        self.human_attention = Attention2(opt, 2)
        self.obj_attention = Attention2(opt)

        self.fc_embed = nn.Sequential(nn.Linear(opt.att_feat_size, opt.rnn_size),
                                      nn.ReLU(),
                                      nn.Dropout(opt.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(opt.att_feat_size),) if opt.use_bn else ()) +
                (nn.Linear(opt.att_feat_size, opt.rnn_size),
                 nn.ReLU(),
                 nn.Dropout(opt.drop_prob_lm)) +
                ((nn.BatchNorm1d(opt.rnn_size),) if opt.use_bn == 2 else ())))

        self.body_embed = nn.Sequential(nn.Linear(opt.fc_feat_size, opt.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(opt.drop_prob_lm))
        self.part_embed = nn.Sequential(nn.Linear(opt.fc_feat_size, opt.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(opt.drop_prob_lm))

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        fc_feats = self.fc_embed(fc_feats)
        # if att_feats is not None:
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation comsumptions.
        # p_att_feats = self.ctx2att(att_feats)
        # else:
        # p_att_feats = None

        return fc_feats, att_feats, att_masks

    def _prepare_body_part_feature(self, body_feat, part_feat):
        body_feat = self.body_embed(body_feat)
        part_feat = self.part_embed(part_feat)
        # hake_fc_feat = self.hake_fc_embed(hake_fc_feat)
        return body_feat, part_feat

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def forward(self, fc_feats, att_feats, body_feats, part_feats, att_masks, body_masks, part_masks):
        fc_feats, att_feats, att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        body_feats, part_feats = self._prepare_body_part_feature(body_feats, part_feats)

        p_fc_feats = fc_feats.unsqueeze(1).expand_as(body_feats)
        p_body_feats = F.relu(self.p_body_fc(body_feats))
        p_body_feats = torch.cat([p_body_feats, body_feats, p_fc_feats], dim=-1)

        part_feats = part_feats.view(part_feats.shape[0], body_feats.shape[1], -1, part_feats.shape[-1])
        pp_fc_feats = fc_feats.view(fc_feats.shape[0], 1, 1, fc_feats.shape[1]).expand_as(part_feats)
        part_feats = torch.cat([part_feats, body_feats.unsqueeze(2).expand_as(part_feats), pp_fc_feats], dim=-1)

        body_feats = self.body_part_to_lantern(p_body_feats)
        part_feats = self.body_part_to_lantern(part_feats)

        part_feats = part_feats * part_masks.view(part_feats.shape[:3]).unsqueeze(-1)

        human_feats = torch.cat([body_feats.unsqueeze(2), part_feats], dim=2)
        human_feats = torch.sum(human_feats, dim=2) / (
                    torch.sum(part_masks.view(part_feats.shape[:3]), dim=2) + 1).unsqueeze(-1)

        p_att_feats = self.ctx2att(att_feats)
        ex_att_feats = att_feats.unsqueeze(1).expand(-1, human_feats.shape[1], -1, -1)
        ex_att_feats = ex_att_feats.contiguous().view(-1, ex_att_feats.shape[-2], ex_att_feats.shape[-1])
        ex_p_att_feats = p_att_feats.unsqueeze(1).expand(-1, human_feats.shape[1], -1, -1)
        ex_p_att_feats = ex_p_att_feats.contiguous().view(-1, ex_p_att_feats.shape[-2], ex_p_att_feats.shape[-1])
        ex_att_masks = att_masks.unsqueeze(1).expand(-1, human_feats.shape[1], -1)
        ex_att_masks = ex_att_masks.contiguous().view(-1, ex_att_masks.shape[-1])

        h_att = human_feats.view(-1, human_feats.shape[-1])
        obj_att, obj_weight = self.obj_attention(h_att, ex_att_feats, ex_p_att_feats, ex_att_masks)
        obj_att = obj_att.view(human_feats.shape)

        tmp_obj_att = F.relu(self.obj_att_fc(obj_att.detach().view(-1, human_feats.shape[-1])))
        h_att = torch.cat([h_att, tmp_obj_att], dim=-1)

        human_att, human_weight = self.human_attention(h_att, ex_att_feats, ex_p_att_feats, ex_att_masks)
        human_att = human_att.view(human_feats.shape)

        # hoi_feats = torch.cat([human_att, obj_att.detach()], dim=-1)
        hoi_feats = human_att

        part_out = self.logit_part(part_feats)
        hoi_out = self.logit_hoi(hoi_feats)
        obj_out = self.logit_obj(obj_att)

        part_out = part_out.view(part_out.shape[0], -1, part_out.shape[-1])
        human_weight = human_weight.view(body_masks.shape[0], body_masks.shape[1], -1)
        obj_weight = obj_weight.view(body_masks.shape[0], body_masks.shape[1], -1)
        human_weight = human_weight * body_masks.unsqueeze(-1)
        obj_weight = obj_weight * body_masks.unsqueeze(-1)

        return hoi_out, part_out, obj_out, human_feats, human_att, obj_att, human_weight, obj_weight


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # features = opt.rnn_size
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.sublayer = PositionwiseFeedForward(size, size, dropout)

    def forward(self, x):
        "Apply residual connection to any sublayer with the same size."
        return self.dropout(self.sublayer(self.norm(x)))

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class imge2sene_fc(nn.Module):
    def __init__(self, opt):
        super(imge2sene_fc, self).__init__()
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm

        self.fc1 = nn.Sequential(nn.Linear(self.rnn_size, self.rnn_size),
                                 nn.LeakyReLU(0.1,inplace=True),
                                 nn.Dropout(self.drop_prob_lm))
        self.fc2 = nn.Sequential(nn.Linear(self.rnn_size, self.rnn_size),
                                 nn.LeakyReLU(0.1, inplace=True),
                                 nn.Dropout(self.drop_prob_lm))
        self.fc3 = nn.Linear(self.rnn_size, self.rnn_size)

    def forward(self, x):
        y1 = self.fc1(x)
        y2 = self.fc2(y1)
        y3 = self.fc3(y2)
        return y3