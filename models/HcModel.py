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
from functools import reduce


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

class HCCM(CaptionModel):
    def __init__(self, opt):
        super(HCCM, self).__init__()
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

        self.att_embed = nn.Sequential(nn.Linear(self.att_feat_size, self.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))


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
        self.core = TopDownCoreHC(opt)

        self.head = HCFH(opt)

        self.weight_thresh = 0.05


        print('create a new mpm_cell')
        mpm_init = np.random.rand(self.rnn_size, self.rnn_size) / 100
        mpm_init = np.float32(mpm_init)
        self.mpm_cell = torch.from_numpy(mpm_init).requires_grad_()


        self.roi2mpm = nn.Sequential(nn.Linear(self.rnn_size, self.rnn_size),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    nn.Dropout(self.drop_prob_lm),
                                    nn.Linear(self.rnn_size, self.rnn_size))



    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def prepare_mpm_feature(self, att_feats):
        self.mpm_cell = self.mpm_cell.cuda(att_feats.get_device())
        att_feats_mpm = self.roi2mpm(att_feats)
        att_shape = att_feats_mpm.shape
        att_feats_mpm = att_feats_mpm.view(-1, att_shape[-1])
        att_feats_mpm = torch.mm(att_feats_mpm, self.mpm_cell)
        att_feats_mpm = att_feats_mpm.view(att_shape)
        p_att_feats_mpm = self.ctx2att(att_feats_mpm)
        return att_feats_mpm, p_att_feats_mpm

    def _prepare_feature(self, activity_feats, back_feats, activity_masks, back_masks, body_feats, body_masks, fc_feats, att_feats, att_masks):

        fc_feats_ori = self.fc_embed(fc_feats)
        att_feats_ori, att_masks_ori = self.clip_att(att_feats, att_masks)
        att_feats_ori = self.att_embed(att_feats_ori)
        p_att_feats_ori = self.ctx2att(att_feats_ori)




        activity_feats = torch.cat([activity_feats,
                                      activity_feats.new_zeros(activity_feats.shape[0],
                                                                 3, activity_feats.shape[-1])], dim=1).contiguous()

        activity_masks = torch.cat([activity_masks,
                                      activity_masks.new_zeros(activity_masks.shape[0],
                                                                 3)], dim=1).contiguous()

        for b_i in range(activity_feats.shape[0]):
            h_index = int((torch.sum(activity_masks[b_i])).item())

            for h_i in range(3):
                if body_masks[b_i, h_i] == 1:
                    activity_feats[b_i, h_index] = body_feats[b_i, h_i]

                    activity_masks[b_i, h_index] = 1

                    h_index += 1
                else:
                    break
        att_feats_activity, att_masks_activity = self.clip_att(activity_feats, activity_masks)
        fc_feats_activity = torch.sum(att_feats_activity, dim=1) / torch.sum(att_masks_activity, dim=1).unsqueeze(-1)
        fc_feats_activity = self.fc_embed(fc_feats_activity)


        att_feats_activity = self.att_embed(att_feats_activity)



        p_att_feats_activity = self.ctx2att(att_feats_activity)



        fc_feats_back = torch.sum(back_feats, dim=1) / (torch.sum(back_masks, dim=1).unsqueeze(-1) + 1e-10)
        fc_feats_back = self.fc_embed(fc_feats_back)

        att_feats_back, att_masks_back = self.clip_att(back_feats, back_masks)

        att_feats_back = self.att_embed(att_feats_back)

        p_att_feats_back = self.ctx2att(att_feats_back)



        fc_feats_all = torch. cat([fc_feats_ori, fc_feats_activity, fc_feats_back], dim=-1)

        return fc_feats_all, att_feats_activity, p_att_feats_activity, att_masks_activity, att_feats_back, \
               p_att_feats_back, att_masks_back, att_feats_ori, p_att_feats_ori, att_masks_ori


    def splitAttFeats(self, att_feats, att_masks, fine_weight, rough_weight):

        fine_weight = fine_weight.view(-1, 3, fine_weight.shape[-1])
        rough_weight = rough_weight.view(-1, 3, fine_weight.shape[-1])
        masks = ((fine_weight > self.weight_thresh) + (rough_weight > self.weight_thresh)) > 0
        tmp = torch.sum(fine_weight + rough_weight, dim=1)
        tmp = ((tmp - tmp.max(dim=-1)[0].unsqueeze(-1)) == 0).float()
        masks = (torch.sum(masks, dim=1).float() + tmp) > 0

        att_feats = att_feats[:, :masks.shape[1]]
        att_masks = att_masks[:, :masks.shape[1]]

        # fine_att = torch.bmm(fine_weight, att_feats)
        # rough_att = torch.bmm(rough_weight, att_feats)


        masks_sum = torch.sum(masks, dim=1)
        max_len = torch.max(masks_sum)

        activity_feats = att_feats.new_zeros(att_feats.shape[0], max_len, att_feats.shape[-1])
        activity_masks = att_masks.new_zeros(att_masks.shape[0], max_len)
        for b_i in range(att_feats.shape[0]):
            activity_feats[b_i, :masks_sum[b_i]] = att_feats[b_i][masks[b_i]]
            activity_masks[b_i, :masks_sum[b_i]] = 1


        masks = 1 - masks
        masks = masks * att_masks.type_as(masks)
        masks_sum = torch.sum(masks, dim=1)
        max_len = torch.max(masks_sum)

        back_feats = att_feats.new_zeros(att_feats.shape[0], max_len, att_feats.shape[-1])
        back_masks = att_masks.new_zeros(att_masks.shape[0], max_len)
        for b_i in range(att_feats.shape[0]):
            back_feats[b_i, :masks_sum[b_i]] = att_feats[b_i][masks[b_i]]
            back_masks[b_i, :masks_sum[b_i]] = 1


        return activity_feats, back_feats, activity_masks, back_masks






    def _forward(self, fc_feats, att_feats, seq, body_feats, part_feats, body_masks, part_masks, att_masks=None):
        '''
        :param fc_feats: (b, d), average value of att_feats
        :param att_feats: (b, n, d), ROI features
        :param seq: (b, l), sentences
        :param body_feats: (b, nh, d), body features
        :param part_feats: (b, nh*10, d), part features
        :param body_masks: (b, nh), if the number of human is nh* in the i-th image , body_masks[i, nh*:] = 0, otherwise 1
        :param part_masks: (b, nh*10), parts of humans shown in the image are set to 1
        :param att_masks: (b, n), ROI: 10 to 100, the positions without ROI features are set to 0, otherwise 1
        :return: word prediction (b, l, vocab_size)
        '''

        # prepare the human feature
        human_feats, fine_weight, rough_weight = self.head(fc_feats, att_feats, body_feats,
                                                           part_feats, att_masks, body_masks,
                                                           part_masks)


        activity_feats, back_feats, activity_masks, back_masks = self.splitAttFeats(att_feats, att_masks, fine_weight, rough_weight)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        # Prepare the features
        fc_feats_all, att_feats_activity, p_att_feats_activity, att_masks_activity, att_feats_back, \
        p_att_feats_back, att_masks_back, att_feats_ori, p_att_feats_ori, att_masks_ori = \
            self._prepare_feature(activity_feats, back_feats, activity_masks, back_masks, body_feats, body_masks, fc_feats, att_feats, att_masks)


        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size + 1)

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

            output, state = self.get_logprobs_state(it, fc_feats_all, att_feats_activity, p_att_feats_activity, att_masks_activity, att_feats_back,
                                                    p_att_feats_back, att_masks_back, att_feats_ori, p_att_feats_ori, att_masks_ori, human_feats, body_masks, state)
            outputs[:, i] = output

        return outputs

    def get_logprobs_state(self, it, fc_feats_all, att_feats_activity, p_att_feats_activity, att_masks_activity, att_feats_back, p_att_feats_back, att_masks_back, att_feats_ori, p_att_feats_ori, att_masks_ori, human_feats, body_masks, state):
        # 'it' contains a word index
        xt = self.embed(it)

        att_feats_mpm, p_att_feats_mpm = self.prepare_mpm_feature(att_feats_ori)
        output, state = self.core(xt, fc_feats_all, att_feats_activity, p_att_feats_activity, att_masks_activity, att_feats_back, p_att_feats_back, att_masks_back, att_feats_ori, p_att_feats_ori, att_masks_ori, att_feats_mpm, p_att_feats_mpm, human_feats, body_masks, state)

        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state

    def _sample_beam(self, fc_feats, att_feats, body_feats, part_feats, body_masks, part_masks, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)



        human_feats, fine_weight, rough_weight = self.head(fc_feats, att_feats, body_feats, part_feats, att_masks,
                                                           body_masks, part_masks)
        activity_feats, back_feats, activity_masks, back_masks = self.splitAttFeats(att_feats, att_masks,
                                                                                          fine_weight, rough_weight)


        fc_feats_all, att_feats_activity, p_att_feats_activity, att_masks_activity, att_feats_back, \
        p_att_feats_back, att_masks_back, att_feats_ori, p_att_feats_ori, att_masks_ori = \
            self._prepare_feature(activity_feats, back_feats, activity_masks, back_masks, body_feats, body_masks,
                                  fc_feats, att_feats, att_masks)



        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats_all = fc_feats_all[k:k+1].expand(beam_size, fc_feats_all.size(1))
            tmp_att_feats_activity = att_feats_activity[k:k+1].expand(*((beam_size,)+att_feats_activity.size()[1:])).contiguous()
            tmp_p_att_feats_activity = p_att_feats_activity[k:k+1].expand(*((beam_size,)+p_att_feats_activity.size()[1:])).contiguous()
            tmp_att_masks_activity = att_masks_activity[k:k+1].expand(*((beam_size,)+att_masks_activity.size()[1:])).contiguous() if att_masks_activity is not None else None

            tmp_att_feats_back = att_feats_back[k:k + 1].expand(*((beam_size,) + att_feats_back.size()[1:])).contiguous()
            tmp_p_att_feats_back = p_att_feats_back[k:k + 1].expand(*((beam_size,) + p_att_feats_back.size()[1:])).contiguous()
            tmp_att_masks_back = att_masks_back[k:k + 1].expand(
                *((beam_size,) + att_masks_back.size()[1:])).contiguous() if att_masks_back is not None else None

            tmp_att_feats_ori = att_feats_ori[k:k + 1].expand(*((beam_size,) + att_feats_ori.size()[1:])).contiguous()
            tmp_p_att_feats_ori = p_att_feats_ori[k:k + 1].expand(*((beam_size,) + p_att_feats_ori.size()[1:])).contiguous()
            tmp_att_masks_ori = att_masks_ori[k:k + 1].expand(
                *((beam_size,) + att_masks_ori.size()[1:])).contiguous() if att_masks_ori is not None else None

            tmp_human_feats = human_feats[k:k + 1].expand(*((beam_size,) + human_feats.size()[1:])).contiguous()

            tmp_body_masks = body_masks[k:k + 1].expand(
                *((beam_size,) + body_masks.size()[1:])).contiguous() if body_masks is not None else None

            for t in range(1):
                if t == 0: # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, state = self.get_logprobs_state(it, tmp_fc_feats_all, tmp_att_feats_activity, tmp_p_att_feats_activity, tmp_att_masks_activity, tmp_att_feats_back,
                                                          tmp_p_att_feats_back, tmp_att_masks_back, tmp_att_feats_ori, tmp_p_att_feats_ori, tmp_att_masks_ori, tmp_human_feats, tmp_body_masks, state)

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats_all, tmp_att_feats_activity, tmp_p_att_feats_activity,
                                                  tmp_att_masks_activity, tmp_att_feats_back, tmp_p_att_feats_back,
                                                  tmp_att_masks_back, tmp_att_feats_ori, tmp_p_att_feats_ori, tmp_att_masks_ori, tmp_human_feats, tmp_body_masks, opt=opt)
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



        human_feats, fine_weight, rough_weight = self.head(fc_feats, att_feats, body_feats, part_feats, att_masks,
                                                            body_masks, part_masks)


        activity_feats, back_feats, activity_masks, back_masks = self.splitAttFeats(att_feats, att_masks,
                                                                                          fine_weight, rough_weight)

        fc_feats_all, att_feats_activity, p_att_feats_activity, att_masks_activity, att_feats_back, \
        p_att_feats_back, att_masks_back, att_feats_ori, p_att_feats_ori, att_masks_ori = \
            self._prepare_feature(activity_feats, back_feats, activity_masks, back_masks, body_feats, body_masks,
                                  fc_feats, att_feats, att_masks)



        trigrams = [] # will be a list of batch_size dictionaries

        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = att_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = att_feats.new_zeros(batch_size, dtype=torch.long)

            logprobs, state = self.get_logprobs_state(it, fc_feats_all, att_feats_activity, p_att_feats_activity,
                                                      att_masks_activity, att_feats_back, p_att_feats_back,
                                                      att_masks_back, att_feats_ori, p_att_feats_ori, att_masks_ori,
                                                      body_masks, state)
            
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



class TopDownCoreHC(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCoreHC, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.rnn_size = opt.rnn_size

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 3, opt.rnn_size) # h^1_t, \hat v

        self.attention_activity = Attention(opt)
        self.attention_back = Attention(opt)

        self.attention_ori = Attention(opt)
        self.attention_mpm = Attention(opt)




        self.human_gate = nn.Sequential(nn.Linear(2*opt.rnn_size, opt.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(opt.drop_prob_lm))
        self.non_gate = nn.Sequential(nn.Linear(2 * opt.rnn_size, opt.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(opt.drop_prob_lm))

        self.part_cali = nn.Sequential(
            nn.Linear(opt.rnn_size + opt.rnn_size, opt.rnn_size),
            nn.ReLU(),
            nn.Dropout(opt.drop_prob_lm),
            nn.Linear(opt.rnn_size, opt.rnn_size)
        )



    def forward(self, xt, fc_feats_all, att_feats_activity, p_att_feats_activity, att_masks_activity, att_feats_back, p_att_feats_back, att_masks_back, att_feats_ori, p_att_feats_ori, att_masks_ori, att_feats_mpm, p_att_feats_mpm, human_feats, body_masks, state):
        prev_h = state[0][-1]

        fc_feats_ori = fc_feats_all[:, :self.rnn_size]
        fc_feats_activity = fc_feats_all[:, self.rnn_size:2 * self.rnn_size]
        fc_feats_back = fc_feats_all[:, 2 * self.rnn_size:3 * self.rnn_size]


        att_lstm_input = torch.cat([prev_h, fc_feats_ori, xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att_activity = self.attention_activity(h_att, att_feats_activity, p_att_feats_activity, att_masks_activity)
        gate_human = torch.sigmoid(self.human_gate(torch.cat([fc_feats_activity, h_att], dim=-1)))

        att_part = torch.sum(human_feats, dim=1) / (torch.sum(body_masks, dim=1).unsqueeze(-1)+1e-10)
        att_part = self.part_cali(torch.cat([att_part, att_activity], dim=-1))

        att_human = att_activity + att_part

        att_obj = self.attention_back(h_att, att_feats_back, p_att_feats_back, att_masks_back)
        gate_obj = torch.sigmoid(self.non_gate(torch.cat([fc_feats_back, h_att], dim=-1)))

        att_ori = self.attention_ori(h_att, att_feats_ori, p_att_feats_ori, att_masks_ori)
        att_ori = att_human * gate_human + att_obj * gate_obj + att_ori

        att_mpm = self.attention_mpm(h_att, att_feats_mpm, p_att_feats_mpm, att_masks_ori)

        lang_lstm_input = torch.cat([att_mpm, att_ori, h_att], 1)

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

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





class HCFH(nn.Module):
    def __init__(self, opt):
        super(HCFH, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.body_part_to_latent = nn.Sequential(
            nn.Linear(3 * self.rnn_size, 2048),
            nn.ReLU(),
            nn.Dropout(opt.drop_prob_lm),
            nn.Linear(2048, self.rnn_size),
            nn.ReLU(),
            nn.Dropout(opt.drop_prob_lm)
        )

        self.ctx2att = nn.Linear(opt.rnn_size, opt.att_hid_size)
        self.p_body_fc = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.obj_att_fc = nn.Linear(opt.rnn_size, opt.rnn_size)

        self.attention_fine = Attention2(opt, 2)
        self.attention_rough = Attention2(opt)

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

        return fc_feats, att_feats, att_masks

    def _prepare_body_part_feature(self, body_feat, part_feat):
        body_feat = self.body_embed(body_feat)
        part_feat = self.part_embed(part_feat)
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

        body_feats = self.body_part_to_latent(p_body_feats)
        part_feats = self.body_part_to_latent(part_feats)

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
        rough_att, rough_weight = self.attention_rough(h_att, ex_att_feats, ex_p_att_feats, ex_att_masks)
        rough_att = rough_att.view(human_feats.shape)

        tmp_rough_att = F.relu(self.obj_att_fc(rough_att.detach().view(-1, human_feats.shape[-1])))
        h_att = torch.cat([h_att, tmp_rough_att], dim=-1)

        fine_att, fine_weight = self.attention_fine(h_att, ex_att_feats, ex_p_att_feats, ex_att_masks)

        fine_weight = fine_weight.view(body_masks.shape[0], body_masks.shape[1], -1)
        rough_weight = rough_weight.view(body_masks.shape[0], body_masks.shape[1], -1)
        fine_weight = fine_weight * body_masks.unsqueeze(-1)
        rough_weight = rough_weight * body_masks.unsqueeze(-1)

        return human_feats, fine_weight, rough_weight

