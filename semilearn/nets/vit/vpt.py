import torch
from torch import nn
from torch.nn import Module

from .vit import vit_tiny_patch2_32, vit_small_patch2_32, vit_small_patch16_224, vit_base_patch16_224, \
    vit_base_patch16_96, VisionTransformer
from ..utils import load_checkpoint


class VPT(VisionTransformer):
    def __init__(self, patch_size=16, embed_dim=768, depth=12, num_heads=12, drop_path_rate=0.2, num_classes=10, **kwargs):
        super().__init__(patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, drop_path_rate=drop_path_rate, num_classes=num_classes)
        self.prompt_length = kwargs['prompt_length']
        self.insert_layers = kwargs['insert_layers']
        if len(self.insert_layers) != 0:
            prompt_shape = (len(self.insert_layers), self.prompt_length, self.embed_dim)
            if kwargs['prompt_init'] == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_shape), requires_grad=True)
                nn.init.uniform_(self.prompt, -1, 1)

    def extract(self, x: torch.Tensor):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for i, block in enumerate(self.blocks):
            j = 0
            if i in self.insert_layers:
                if i == self.insert_layers[0]:
                    x = torch.cat([x[:, 0, :].unsqueeze(1),
                                   self.prompt[j, ...].squeeze(0).expand(x.shape[0], -1, -1),
                                   x[:, 1:, :]], dim=1)
                else:
                    x = torch.cat([x[:, 0, :].unsqueeze(1),
                                   self.prompt[j, ...].squeeze(0).expand(x.shape[0], -1, -1),
                                   x[:, (1 + self.prompt_length):, :]], dim=1)
                j += 1
            x = self.blocks[i](x)
        x = self.norm(x)
        return x

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        if hasattr(self, 'prompt'):
            self.prompt.requires_grad = True
        for param in self.head.parameters():
            param.requires_grad = True


def vpt_on_vit_base_patch_16_224(pretrained=True, pretrained_path='/home/lhz/code/Semi-Supervised-Learning/vit_base_patch16_224.augreg2_in21k_ft_in1k/pytorch_model.bin', num_classes=1000):

    # insert_layers: 0 means shallow pt, [] means finetune a classifier
    model_kwargs = dict(prompt_length=12, prompt_init='uniform', insert_layers=[0])
    model = VPT(patch_size=16, embed_dim=768, depth=12, num_heads=12, drop_path_rate=0.2, num_classes=num_classes, **model_kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    model.freeze()
    return model


def ftcls_on_vit_base_patch_16_224(pretrained=True, pretrained_path='/home/lhz/code/Semi-Supervised-Learning/vit_base_patch16_224.augreg2_in21k_ft_in1k/pytorch_model.bin', num_classes=1000):

    # insert_layers: 0 means shallow pt, [] means finetune a classifier
    model_kwargs = dict(prompt_length=12, prompt_init='uniform', insert_layers=[])
    model = VPT(patch_size=16, embed_dim=768, depth=12, num_heads=12, drop_path_rate=0.2, num_classes=num_classes, **model_kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    model.freeze()
    return model
