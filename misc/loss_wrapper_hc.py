import torch
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
import json


# def merge_sent(seq):
#     tmp = seq.new_zeros((seq.shape[0] / 2, seq.shape[1]))
#     for i in range(seq.shape[0] / 2):
#         index_start = 0
#         for j in range(seq.shape[1]):
#             if seq[i * 2, j] == 0:
#                 break
#             else:
#                 index_start += 1
#         for j in range(index_start):
#             tmp[i, j] = seq[i * 2, index_start - j - 1]
#         # index_end = 0
#         for j in range(seq.shape[1]):
#             if seq[i * 2 + 1, j] == 0:
#                 break
#             else:
#                 if index_start != 0:
#                     tmp[i, j] = seq[i * 2 + 1, j]
#                 else:
#                     if j + index_start - 1 == seq.shape[1]:
#                         break
#                     tmp[i, j + index_start - 1] = seq[i * 2 + 1, j]
#     return tmp

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            self.crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = utils.LanguageModelCriterion()

        if opt.enable_no_interaction == 1:
            counts = json.load(open('data/part_hoi_class_count.json', 'r'))
            hoi_count = counts['hoi_count']
            part_state_count = counts['part_state_count']
            hoi_pos_weight = torch.Tensor(hoi_count)
            part_state_pos_weight = torch.Tensor(part_state_count)
            hoi_pos_weight = 1.0 / (hoi_pos_weight / torch.min(hoi_pos_weight))
            part_state_pos_weight = 1.0 / (part_state_pos_weight / torch.min(part_state_pos_weight))
            hoi_pos_weight = hoi_pos_weight / torch.min(hoi_pos_weight) / 50.0
            part_state_pos_weight = part_state_pos_weight / torch.min(part_state_pos_weight) / 50.0
            hoi_pos_weight = hoi_pos_weight.unsqueeze(0)
            part_state_pos_weight = part_state_pos_weight.unsqueeze(0)
            self.hoi_crit = utils.ClassificationCriterionBCE(hoi_pos_weight)
            self.part_state_crit = utils.ClassificationCriterionBCE(part_state_pos_weight)
        else:
            self.hoi_crit = utils.ClassificationCriterionBCE()
            self.part_state_crit = utils.ClassificationCriterionBCE()


        self.obj_crit = utils.ClassificationCriterion()


        self.rl_crit = utils.RewardCriterion()



    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices, sc_flag, body_feats, part_feats, body_masks, part_masks):
        out = {}
        if not sc_flag:
            loss = self.crit(self.model(fc_feats, att_feats, labels, body_feats, part_feats, body_masks, part_masks, att_masks), labels[:,1:], masks[:,1:])
        else:
            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model(fc_feats, att_feats, body_feats, part_feats, body_masks, part_masks, att_masks, mode='sample')
            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, body_feats, part_feats, body_masks, part_masks, att_masks, opt={'sample_method':'sample'}, mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            # greedy_res_merge = merge_sent(greedy_res)
            # gen_result_merge = merge_sent(gen_result)
            # reward = get_self_critical_reward(greedy_res_merge, gts, gen_result_merge, self.opt)
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = torch.from_numpy(reward).float().to(gen_result.device)
            # reward = reward.unsqueeze(1).expand(-1, 2)
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
            out['reward'] = reward[:,0].mean()


        out['loss'] = loss
        return out
