import torch
from torch import nn
from torch.nn import Module

from .vit import vit_tiny_patch2_32, vit_small_patch2_32, vit_small_patch16_224, vit_base_patch16_224, \
    vit_base_patch16_96, VisionTransformer
from ..utils import load_checkpoint


class DualPrompt(VisionTransformer):
    def __init__(self, patch_size=16, embed_dim=768, depth=12, num_heads=12, drop_path_rate=0.2, num_classes=10, **kwargs):
        super().__init__(patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, drop_path_rate=drop_path_rate, num_classes=num_classes)
        self.prompt_length = kwargs['prompt_length']
        self.insert_layers = kwargs['insert_layers']

        prompt_shape = (len(self.insert_layers), self.prompt_length, self.embed_dim)
        if kwargs['prompt_init'] == 'uniform':
            self.simclr_prompt = nn.Parameter(torch.randn(prompt_shape), requires_grad=True)
            nn.init.uniform_(self.simclr_prompt, -1, 1)
            self.ce_prompt = nn.Parameter(torch.randn(prompt_shape), requires_grad=True)
            nn.init.uniform_(self.ce_prompt, -1, 1)

        self.projector = nn.Linear(768, 128)

    def extract(self, x: torch.Tensor, is_ce=True):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        if is_ce:
            prompt = self.ce_prompt
        else:
            prompt = self.simclr_prompt
        for i, block in enumerate(self.blocks):
            j = 0
            if i in self.insert_layers:
                if i == self.insert_layers[0]:
                    x = torch.cat([x[:, 0, :].unsqueeze(1),
                                   prompt[j, ...].squeeze(0).expand(x.shape[0], -1, -1),
                                   x[:, 1:, :]], dim=1)
                else:
                    x = torch.cat([x[:, 0, :].unsqueeze(1),
                                   prompt[j, ...].squeeze(0).expand(x.shape[0], -1, -1),
                                   x[:, (1 + self.prompt_length):, :]], dim=1)
                j += 1
            x = self.blocks[i](x)
        x = self.norm(x)
        return x

    def forward(self, x, only_fc=False, only_feat=False, is_ce=True, projected=False, **kwargs):
        if only_fc:
            return self.head(x)
        x = self.extract(x, is_ce=is_ce)
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)

        if only_feat:
            if projected:
                x = self.projector(x)
            return x

        output = self.head(x)
        result_dict = {'logits': output, 'feat': x}
        return result_dict

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.ce_prompt.requires_grad = True
        self.simclr_prompt.requires_grad = True
        for param in self.head.parameters():
            param.requires_grad = True
        for param in self.projector.parameters():
            param.requires_grad = True


def dualpt_on_vit_base_patch_16_224(pretrained=True, pretrained_path='/home/lhz/code/semi-pt/vit_base_patch16_224'
                                                                  '.augreg2_in21k_ft_in1k/pytorch_model.bin', num_classes=1000):

    # insert_layers: 0 means shallow pt, [] means finetune a classifier
    model_kwargs = dict(prompt_length=12, prompt_init='uniform', insert_layers=[0])
    model = DualPrompt(patch_size=16, embed_dim=768, depth=12, num_heads=12, drop_path_rate=0.2, num_classes=num_classes, **model_kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)

    return model
