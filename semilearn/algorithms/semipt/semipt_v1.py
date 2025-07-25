import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, top_k_accuracy_score, balanced_accuracy_score, precision_score, \
    recall_score, f1_score, confusion_matrix
import torch.nn.functional as F

from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.criterions import NTxentLoss, L2SimilarityLoss
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool


# @ALGORITHMS.register('semipt_v1')
class SemiPT(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # fixmatch specified arguments
        self.samples_count = 0
        self.labeled_count = 0
        self.correct_count = 0
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)

    def init(self, T, p_cutoff, hard_label=True):
        self.T = T
        self.NT_xent_loss = NTxentLoss(self.gpu, self.T)
        self.l2_similarity_loss = L2SimilarityLoss()
        self.use_hard_label = hard_label
        self.p_cutoff = p_cutoff
        self.change_state = 0

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def train_step(self, x_lb_w, x_lb_s_1, x_lb_s_2, x_lb_m_1, x_lb_m_2, y_lb, x_ulb_w, x_ulb_s_0, x_ulb_s_1, y_ulb):
        # phase one, only use unlabeled data to calculate unsup loss and update SimCLR prompt
        if self.epoch < self.epochs // 2:
            feats_x_ulb_s_0 = self.model(x_ulb_s_0, only_feat=True, projected=True, is_ce=False)
            # feats_x_ulb_w = self.model(x_ulb_w, only_feat=True, projected=True, no_prompt=True)
            feats_x_ulb_s_1 = self.model(x_ulb_s_1, only_feat=True, projected=True, is_ce=False)
            unsup_loss = self.NT_xent_loss(feats_x_ulb_s_0, feats_x_ulb_s_1)
            # unsup_loss = self.NT_xent_loss(feats_x_ulb_w, feats_x_ulb_s_1)

            outs_x_lb_sim = self.model(x_lb_w, only_feat=False, projected=False, is_ce=False)
            logits_x_lb_sim = outs_x_lb_sim['logits']
            feats_x_lb_sim = outs_x_lb_sim['feat']
            sup_loss = self.ce_loss(logits_x_lb_sim, y_lb, reduction='mean')

            feat_dict = {'x_ulb_s_0': feats_x_ulb_s_0, 'x_ulb_s_1': feats_x_ulb_s_1, 'x_lb_w': feats_x_lb_sim}
            # feat_dict = {'x_ulb_w': feats_x_ulb_w, 'x_ulb_s_1': feats_x_ulb_s_1, 'x_lb_w': feats_x_lb_sim}
            total_loss = sup_loss + self.lambda_u * unsup_loss
            out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
            log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                             unsup_loss=unsup_loss.item(),
                                             total_loss=total_loss.item())
        # phase two, use labeled data to train another prompt, and use the prompts learned in the
        # first stage to constrain the updates of this prompt
        else:
            if self.change_state == 0:
                self.load_best_model(os.path.join(self.save_dir, self.save_name, 'model_best.pth'))
                self.model.freeze()
                # reset optimizer and scheduler
                self.optimizer, self.scheduler = self.set_optimizer()
                self.change_state = 1

            with torch.no_grad():
                outs_x_ulb_sim = self.model(x_ulb_w, only_feat=False, projected=False, is_ce=False)
                logits_x_ulb_sim = outs_x_ulb_sim['logits']
                feats_x_ulb_sim = outs_x_ulb_sim['feat']
            probs_x_ulb = self.compute_prob(logits_x_ulb_sim.detach())
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb, softmax_x_ulb=False)
            pseudo_label = self.call_hook('gen_ulb_targets', "PseudoLabelingHook",
                                          logits=probs_x_ulb,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)
            self.samples_count += pseudo_label.shape[0]
            self.correct_count += torch.sum(torch.argmax(probs_x_ulb, dim=-1) == y_ulb).item()
            self.labeled_count += torch.sum(mask).item()
            if (self.it + 1) % 1024 == 0:
                self.print_fn(self.samples_count)
                self.print_fn(self.correct_count)
                self.print_fn(self.labeled_count)

            outs_x_ulb_ce = self.model(x_ulb_w, only_feat=False, projected=False, is_ce=True)
            logits_x_ulb_ce = outs_x_ulb_ce['logits']
            feats_x_ulb_ce = outs_x_ulb_ce['feat']
            outs_x_lb = self.model(x_lb_w, only_feat=False, projected=False, is_ce=True)
            logit_x_lb = outs_x_lb['logits']
            feat_x_lb = outs_x_lb['feat']
            feat_dict = {'x_lb': feat_x_lb, 'x_ulb_sim': feats_x_ulb_sim, 'x_ulb_ce': feats_x_ulb_ce}

            simclr_prompt = self.model.simclr_prompt.detach()
            sup_loss = self.ce_loss(logit_x_lb, y_lb, reduction='mean') + self.consistency_loss(logits_x_ulb_ce, pseudo_label, 'ce', mask=mask)
            # kl_loss = self.consistency_loss(logits_x_lb, logits_x_lb_original, name='kl')
            unsup_loss = self.l2_similarity_loss(self.model.ce_prompt, simclr_prompt)
            # unsup_loss = self.consistency_loss(logits_x_ulb_ce, pseudo_label, 'ce', mask=mask)
            # total_loss = sup_loss
            total_loss = sup_loss + self.lambda_u * unsup_loss
            # total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_u * kl_loss
            # out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
            out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
            log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                             unsup_loss=unsup_loss.item(),
                                             total_loss=total_loss.item())
        return out_dict, log_dict

    def evaluate(self, eval_dest='eval', out_key='logits', return_logits=False):
        # we won't evaluate the model when it comes to learn in an unsupervised state
        # if self.epoch < self.epochs // 2:
        #     eval_dict = {eval_dest + '/loss': 0, eval_dest + '/top-1-acc': 0,
        #                  eval_dest + '/top-5-acc': 0,
        #                  eval_dest + '/balanced_acc': 0, eval_dest + '/precision': 0,
        #                  eval_dest + '/recall': 0, eval_dest + '/F1': 0}
        #     if return_logits:
        #         eval_dict[eval_dest + '/logits'] = 0
        #     return eval_dict
        # else:
        self.model.eval()
        self.ema.apply_shadow()

        eval_loader = self.loader_dict[eval_dest]
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_probs = []
        y_logits = []
        with torch.no_grad():
            for data in eval_loader:
                x = data['x_lb']
                y = data['y_lb']

                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                num_batch = y.shape[0]
                total_num += num_batch
                if self.epoch < self.epochs // 2:
                    logits = self.model(x, is_ce=False)[out_key]
                else:
                    logits = self.model(x)[out_key]
                loss = F.cross_entropy(logits, y, reduction='mean', ignore_index=-1)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                y_logits.append(logits.cpu().numpy())
                y_probs.extend(torch.softmax(logits, dim=-1).cpu().tolist())
                total_loss += loss.item() * num_batch
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_logits = np.concatenate(y_logits)
        top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_probs, k=5)
        balanced_top1 = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        F1 = f1_score(y_true, y_pred, average='macro')

        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()

        eval_dict = {eval_dest + '/loss': total_loss / total_num, eval_dest + '/top-1-acc': top1,
                     eval_dest + '/top-5-acc': top5,
                     eval_dest + '/balanced_acc': balanced_top1, eval_dest + '/precision': precision,
                     eval_dest + '/recall': recall, eval_dest + '/F1': F1}
        if return_logits:
            eval_dict[eval_dest + '/logits'] = y_logits
        return eval_dict

    def load_best_model(self, load_path):
        checkpoint = torch.load(load_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.it = 5120
        self.start_epoch = 5
        self.epoch = 5
        self.print_fn('Model loaded')
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--T', float, 0.5),
        ]
