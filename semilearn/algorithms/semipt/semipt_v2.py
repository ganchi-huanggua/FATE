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


# @ALGORITHMS.register('semipt_v2')
class SemiPT(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # fixmatch specified arguments
        self.correct_labeled_count = 0
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

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s_0, x_ulb_s_1, y_ulb):
        # phase one, only use unlabeled data to calculate unsup loss and update SimCLR prompt
        if self.epoch < 10:
            feats_x_ulb_s_0 = self.model(x_ulb_s_0, only_feat=True, projected=True, simclr=True, ce=False)
            feats_x_ulb_s_1 = self.model(x_ulb_s_1, only_feat=True, projected=True, simclr=True, ce=False)
            unsup_loss = self.NT_xent_loss(feats_x_ulb_s_0, feats_x_ulb_s_1)

            feat_dict = {'x_ulb_s_0': feats_x_ulb_s_0, 'x_ulb_s_1': feats_x_ulb_s_1}
            total_loss = unsup_loss
            out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
            log_dict = self.process_log_dict(unsup_loss=unsup_loss.item(),
                                             total_loss=total_loss.item())
        # phase two, use labeled data to train another prompt, and use the prompts learned in the
        # first stage to constrain the updates of this prompt
        elif 10 <= self.epoch < 60:
            if self.change_state != 1:
                self.model.simclr_prompt.requires_grad = False
                # reset optimizer and scheduler
                self.optimizer, self.scheduler = self.set_optimizer()
                self.change_state = 1
            # upperbound: 64.24
            outs_x_ulb_s = self.model(x_ulb_s_0, only_feat=False, projected=False, ce=True, simclr=False)
            logits_x_ulb_s = outs_x_ulb_s['logits']
            feats_x_ulb_s = outs_x_ulb_s['feat']

            with torch.no_grad():
                outs_x_ulb_w = self.model(x_ulb_w, only_feat=False, projected=False, ce=True, simclr=True)
                logits_x_ulb_w = outs_x_ulb_w['logits']
                feats_x_ulb_w = outs_x_ulb_w['feat']

            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)
            self.samples_count += pseudo_label.shape[0]
            self.correct_count += torch.sum(torch.argmax(probs_x_ulb_w, dim=-1) == y_ulb).item()
            self.labeled_count += torch.sum(mask).item()
            self.correct_labeled_count += torch.sum(
                ((torch.argmax(probs_x_ulb_w, dim=-1) == y_ulb) == 1) & (mask == 1)).item()
            if (self.it + 1) % 1024 == 0:
                self.print_fn(self.samples_count)
                self.print_fn(self.correct_count)
                self.print_fn(self.labeled_count)
                self.print_fn(self.correct_labeled_count)

            outs_x_lb = self.model(x_lb, only_feat=False, projected=False, ce=True, simclr=True)
            logits_x_lb = outs_x_lb['logits']
            feats_x_lb = outs_x_lb['feat']
            outs_x_lb_1 = self.model(x_lb, only_feat=False, projected=False, ce=True, simclr=False)
            logits_x_lb_1 = outs_x_lb_1['logits']
            feats_x_lb_1 = outs_x_lb_1['feat']
            feat_dict = {}
            unsup_loss = self.consistency_loss(logits_x_ulb_s, pseudo_label, 'ce', mask=mask)
            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean') + self.ce_loss(logits_x_lb_1, y_lb, reduction='mean')
            # simclr_prompt = self.model.simclr_prompt.detach()
            # unsup_loss = self.l2_similarity_loss(self.model.ce_prompt, simclr_prompt)
            total_loss = sup_loss + unsup_loss
            # total_loss = unsup_loss
            out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
            log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                             unsup_loss=unsup_loss.item(),
                                             total_loss=total_loss.item(),
                                             util_ratio=mask.float().mean().item())
            # outs_x_lb_0 = self.model(x_lb, only_feat=False, projected=False, simclr=True, ce=False)
            # logits_x_lb_0 = outs_x_lb_0['logits']
            # feats_x_lb_0 = outs_x_lb_0['feat']
            # outs_x_lb_1 = self.model(x_lb, only_feat=False, projected=False, simclr=False, ce=False)
            # logits_x_lb_1 = outs_x_lb_1['logits']
            # feats_x_lb_1 = outs_x_lb_1['feat']
            #
            # sup_loss = self.ce_loss(logits_x_lb_0, y_lb, reduction='mean') + self.ce_loss(logits_x_lb_1, y_lb, reduction='mean')
            # total_loss = sup_loss
            # # feat_dict = {'x_lb_w': feats_x_lb}
            # feat_dict = {}
            # out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
            # log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
            #                                  total_loss=total_loss.item())
        else:
            if self.change_state != 2:
                self.load_best_model(os.path.join(self.save_dir, self.save_name, 'model_best.pth'))
                # self.model.freeze()
                self.model.simclr_prompt.requires_grad = True
                # for param in self.model.simclr_head.parameters():
                #     param.requires_grad = False
                self.optimizer, self.scheduler = self.set_optimizer()
                self.change_state = 2

            outs_x_ulb_s = self.model(x_ulb_s_0, only_feat=False, projected=False, is_ce=False)
            logits_x_ulb_s = outs_x_ulb_s['logits']
            feats_x_ulb_s = outs_x_ulb_s['feat']

            with torch.no_grad():
                outs_x_ulb_w = self.model(x_ulb_w, only_feat=False, projected=False, is_ce=False)
                logits_x_ulb_w = outs_x_ulb_w['logits']
                feats_x_ulb_w = outs_x_ulb_w['feat']

            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)
            self.samples_count += pseudo_label.shape[0]
            self.correct_count += torch.sum(torch.argmax(probs_x_ulb_w, dim=-1) == y_ulb).item()
            self.labeled_count += torch.sum(mask).item()
            self.correct_labeled_count += torch.sum(((torch.argmax(probs_x_ulb_w, dim=-1) == y_ulb) == 1) & (mask == 1)).item()
            if (self.it + 1) % 1024 == 0:
                self.print_fn(self.samples_count)
                self.print_fn(self.correct_count)
                self.print_fn(self.labeled_count)
                self.print_fn(self.correct_labeled_count)

            outs_x_lb = self.model(x_lb, only_feat=False, projected=False, is_ce=False)
            logits_x_lb = outs_x_lb['logits']
            feats_x_lb = outs_x_lb['feat']
            feat_dict = {}
            unsup_loss = self.consistency_loss(logits_x_ulb_s, pseudo_label, 'ce', mask=mask)
            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            # simclr_prompt = self.model.simclr_prompt.detach()
            # unsup_loss = self.l2_similarity_loss(self.model.ce_prompt, simclr_prompt)
            total_loss = sup_loss + unsup_loss
            # total_loss = unsup_loss
            out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
            log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                             unsup_loss=unsup_loss.item(),
                                             total_loss=total_loss.item(),
                                             util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    def evaluate(self, eval_dest='eval', out_key='logits', return_logits=False):
        # we won't evaluate the model when it comes to learn in an unsupervised state
        if self.change_state == 0:
            eval_dict = {eval_dest + '/loss': 0, eval_dest + '/top-1-acc': 0,
                         eval_dest + '/top-5-acc': 0,
                         eval_dest + '/balanced_acc': 0, eval_dest + '/precision': 0,
                         eval_dest + '/recall': 0, eval_dest + '/F1': 0}
            if return_logits:
                eval_dict[eval_dest + '/logits'] = 0
            return eval_dict
        else:
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
                    if self.change_state == 1:
                        logits = self.model(x, simclr=True, ce=True)[out_key]
                    else:
                        logits = self.model(x, is_ce=False)[out_key]
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
        match = self.model.load_state_dict(checkpoint['model'], strict=False)
        self.ema_model.load_state_dict(checkpoint['ema_model'], strict=False)
        self.loss_scaler.load_state_dict(checkpoint['loss_scaler'])
        # self.optimizer.load_state_dict(checkpoint['optimizer'])
        # if self.scheduler is not None and 'scheduler' in checkpoint:
        #     self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = 20480
        self.start_epoch = 20
        self.epoch = 20
        self.print_fn('Model loaded')
        self.print_fn(match)
        return checkpoint

    def load_model(self, load_path):
        """
        load model and specified parameters for resume
        """
        checkpoint = torch.load(load_path, map_location='cpu')
        self.it = checkpoint['it']
        self.start_epoch = checkpoint['epoch']
        self.epoch = self.start_epoch
        self.best_it = checkpoint['best_it']
        self.best_eval_acc = checkpoint['best_eval_acc']
        if self.it != 10240:
            match = self.model.load_state_dict(checkpoint['model'])
            self.ema_model.load_state_dict(checkpoint['ema_model'])
            self.loss_scaler.load_state_dict(checkpoint['loss_scaler'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.scheduler is not None and 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            if self.epoch >= 10:
                self.model.simclr_prompt.requires_grad = False
                self.change_state = 1

        else:
            pretrained_dict = {k: v for k, v in checkpoint['model'].items() if 'simclr_prompt' in k}
            match = self.model.load_state_dict(pretrained_dict, strict=False)

        self.print_fn('Model loaded')
        self.print_fn(match)
        return checkpoint


    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--T', float, 0.5),
        ]
