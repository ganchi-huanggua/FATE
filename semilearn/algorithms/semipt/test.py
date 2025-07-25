import torch
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.criterions import NTxentLoss, L2SimilarityLoss
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool


@ALGORITHMS.register('testsemipt')
class TestSemiPT(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # fixmatch specified arguments
        self.init(T=args.T, phase_2_epoch=self.epochs // 2, )

    def init(self, T, phase_2_epoch):
        self.T = T
        self.phase_2_epoch = phase_2_epoch
        self.NT_xent_loss = NTxentLoss(self.gpu, self.T)
        self.l2_similarity_loss = L2SimilarityLoss()
        self.change_state = 0

    def set_hooks(self):
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def train_step(self, x_lb_w, x_lb_s_1, x_lb_s_2, x_lb_m_1, x_lb_m_2, y_lb, x_ulb_s_0, x_ulb_s_1):
        # phase one, only use unlabeled data to calculate unsup loss and update SimCLR prompt
        if self.epoch < self.epochs // 2:
            feats_x_ulb_s_0 = self.model(x_ulb_s_0, only_feat=True, projected=True, is_ce=False)
            feats_x_ulb_s_1 = self.model(x_ulb_s_1, only_feat=True, projected=True, is_ce=False)
            unsup_loss = self.NT_xent_loss(feats_x_ulb_s_0, feats_x_ulb_s_1)
            feat_dict = {'x_ulb_s_0': feats_x_ulb_s_0, 'x_ulb_s_1': feats_x_ulb_s_1}
            total_loss = unsup_loss
            out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
            log_dict = self.process_log_dict(sup_loss=0,
                                             unsup_loss=unsup_loss.item(),
                                             total_loss=total_loss.item())
        # phase two, use labeled data to train another prompt, and use the prompts learned in the
        # first stage to constrain the updates of this prompt
        else:
            if self.change_state == 0:
                # reset optimizer and scheduler
                self.optimizer, self.scheduler = self.set_optimizer()
                self.change_state = 1

            outs_x_lb = self.model(x_lb_w, only_feat=False, projected=False, is_ce=True)
            logit_x_lb = outs_x_lb['logits']
            feat_x_lb = outs_x_lb['feat']
            feat_dict = {'x_lb': feat_x_lb}
            sup_loss = self.ce_loss(logit_x_lb, y_lb, reduction='mean')
            # kl_loss = self.consistency_loss(logits_x_lb, logits_x_lb_original, name='kl')
            simclr_prompt = self.model.simclr_prompt.detach()
            unsup_loss = self.l2_similarity_loss(self.model.ce_prompt, simclr_prompt)
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
        if self.epoch < self.epochs // 2:
            eval_dict = {eval_dest + '/loss': 0, eval_dest + '/top-1-acc': 0,
                         eval_dest + '/top-5-acc': 0,
                         eval_dest + '/balanced_acc': 0, eval_dest + '/precision': 0,
                         eval_dest + '/recall': 0, eval_dest + '/F1': 0}
            if return_logits:
                eval_dict[eval_dest + '/logits'] = 0
            return eval_dict
        else:
            return super().evaluate(eval_dest, out_key, return_logits)

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--T', float, 0.5),
        ]