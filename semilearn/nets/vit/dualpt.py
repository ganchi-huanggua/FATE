import cv2
import numpy as np
import torch
import torchvision.datasets
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from .vit import VisionTransformer
from ..utils import load_checkpoint
# from vit import VisionTransformer
# from .utils import load_checkpoint

class DualPrompt(VisionTransformer):
    def __init__(self, patch_size=16, embed_dim=768, depth=12, num_heads=12, drop_path_rate=0.2, num_classes=10,
                 **kwargs):
        super().__init__(patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                         drop_path_rate=drop_path_rate, num_classes=num_classes)
        self.prompt_length = kwargs['prompt_length']
        self.insert_layers = kwargs['insert_layers']

        prompt_shape = (len(self.insert_layers), self.prompt_length, self.embed_dim)
        if kwargs['prompt_init'] == 'uniform':
            self.simclr_prompt = nn.Parameter(torch.randn(prompt_shape), requires_grad=True)
            nn.init.uniform_(self.simclr_prompt, -1, 1)
            # self.ce_prompt = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)  # new cls token: 1, 1, 768
            self.ce_prompt = nn.Parameter(torch.randn(prompt_shape), requires_grad=True)
            nn.init.uniform_(self.ce_prompt, -1, 1)
        self.projector = nn.Linear(768, 128)
        self.simclr_head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.ce_head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.noise_prompt = None

    def extract(self, x: torch.Tensor, simclr=False, ce=False, noise=False):

        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        # if noise:
        #     if self.noise_prompt is None:
        #         noise_prompt = nn.Parameter(torch.randn(1, 12, 768), requires_grad=False).to(x.device)
        #         nn.init.uniform_(noise_prompt, -1, 1)
        #         # noise_prompt = nn.Parameter(torch.zeros(1, 12, 768), requires_grad=False).to(x.device)
        #         self.noise_prompt = noise_prompt.detach()

        if simclr:
            x = torch.cat([x[:, 0, :].unsqueeze(1),
                           self.simclr_prompt[0, ...].squeeze(0).expand(x.shape[0], -1, -1),
                           x[:, 1:, :]], dim=1)
        if ce:
            # x = torch.cat((x, self.ce_prompt.expand(x.shape[0], -1, -1)), dim=1)
            x = torch.cat([x, self.ce_prompt[0, ...].squeeze(0).expand(x.shape[0], -1, -1)], dim=1)
        # [1 (original cls token), 12 (simclr prompt), 196 (patched embedding), 1 (ce prompt / new cls token)] so far
        # if noise:
        #     x = torch.cat([x[:, 0, :].unsqueeze(1),
        #                    self.noise_prompt[0, ...].squeeze(0).expand(x.shape[0], -1, -1),
        #                    x[:, 1:, :]], dim=1)
        # print(x.shape)
        for i, block in enumerate(self.blocks):
            # if i == 7 and noise:
            #     x = torch.cat([x[:, 0, :].unsqueeze(1),
            #                    self.noise_prompt[0, ...].squeeze(0).expand(x.shape[0], -1, -1),
            #                    x[:, 1:, :]], dim=1)
            x = self.blocks[i](x)
        x = self.norm(x)
        return x

    def forward(self, x, only_feat=False, simclr=False, ce=False, projected=False, noise=False, **kwargs):

        x = self.extract(x, simclr=simclr, ce=ce, noise=noise)
        # if simclr and not ce:
        #     self.global_pool = 'token'
        # if simclr and ce:
        #     self.global_pool = 'new_token_mapped'
        if ce:
            self.global_pool = 'new_token_mapped'
        else:
            self.global_pool = 'token'

        # fixme: ablation
        # self.global_pool = 'token'
        if self.global_pool == 'avg':
            x = x[:, 1:].mean(dim=1)
        elif self.global_pool == 'token':
            x = x[:, 0]
        elif self.global_pool == 'new_token_mapped':
            x = x[:, -self.prompt_length:].mean(dim=1)

        x = self.fc_norm(x)

        if only_feat:
            if projected:
                x = self.projector(x)
            return x

        output = self.simclr_head(x)
        result_dict = {'logits': output, 'feat': x}
        return result_dict

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.ce_prompt.requires_grad = True
        self.simclr_prompt.requires_grad = True
        for param in self.ce_head.parameters():
            param.requires_grad = True
        for param in self.simclr_head.parameters():
            param.requires_grad = True
        # for param in self.head.parameters():
        #     param.requires_grad = True
        for param in self.projector.parameters():
            param.requires_grad = True

        # fixme:
        # self.simclr_prompt.requires_grad = False

    def freeze_exp(self):
        self.freeze()


def dualpt_on_vit_base_patch_16_224(pretrained=True, pretrained_path="/home/lhz/code/Semi-Supervised-Learning/vit_base_patch16_224.augreg2_in21k_ft_in1k/pytorch_model.bin",
                                    num_classes=1000):
    # insert_layers: 0 means shallow pt, [] means finetune a classifier
    model_kwargs = dict(prompt_length=12, prompt_init='uniform', insert_layers=[0])
    model = DualPrompt(patch_size=16, embed_dim=768, depth=12, num_heads=12, drop_path_rate=0.2,
                       num_classes=num_classes, **model_kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    model.freeze()
    return model


if __name__ == "__main__":
    model_kwargs = dict(prompt_length=12, prompt_init='uniform', insert_layers=[0])
    model = DualPrompt(patch_size=16, embed_dim=768, depth=12, num_heads=12, drop_path_rate=0.2,
                       num_classes=200, **model_kwargs)
    pretrained_path = "/home/lhz/code/Semi-Supervised-Learning/saved_models/classic_cv/semipt_final_cub/model_best.pth"
    data_path = "/home/lhz/data/cifar10"
    model = load_checkpoint(model, pretrained_path)
    for param in model.parameters():
        param.requires_grad = True
    torch.cuda.set_device(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_origin = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    cifar10 = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=transform)
    cifar10_origin = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=transform_origin)
    print(cifar10.class_to_idx)
    loader = DataLoader(cifar10, batch_size=64, shuffle=False, num_workers=16)

    # img, label = cifar10[0]
    # img = img.unsqueeze(0)  # 添加batch维度
    # img = img.to(device)
    # input_tensor = img
    # img_origin = cifar10_origin[0][0].unsqueeze(0)
    img = Image.open("/home/lhz/data/CUB_200_2011/images/200.Common_Yellowthroat/Common_Yellowthroat_0028_190527.jpg")
    img = img.resize((224, 224))
    input_tensor = transform(img).unsqueeze(0).to(device)


    def rollout(attentions, discard_ratio, head_fusion="max"):
        result = torch.eye(attentions[0].size(-1))
        with torch.no_grad():
            for attention in attentions:
                if head_fusion == "mean":
                    attention_heads_fused = attention.mean(axis=1)
                elif head_fusion == "max":
                    attention_heads_fused = attention.max(axis=1)[0]
                elif head_fusion == "min":
                    attention_heads_fused = attention.min(axis=1)[0]
                else:
                    raise "Attention head fusion type Not supported"

                # Drop the lowest attentions, but
                # don't drop the class token
                flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
                indices = indices[indices != 0]
                flat[0, indices] = 0

                I = torch.eye(attention_heads_fused.size(-1))
                a = (attention_heads_fused + 1.0 * I) / 2
                a = a / a.sum(dim=-1)

                result = torch.matmul(a, result)

        # Look at the total attention between the class token,
        # and the image patches
        mask = result[0, 0, 1:]
        # mask = mask[12:208]
        # In case of 224x224 image, this brings us from 196 to 14
        width = int(mask.size(-1) ** 0.5)
        mask = mask.reshape(width, width).numpy()
        mask = mask / np.max(mask)
        return mask


    class VITAttentionRollout:
        def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean",
                     discard_ratio=0.9):
            self.model = model
            self.head_fusion = head_fusion
            self.discard_ratio = discard_ratio
            for name, module in self.model.named_modules():
                if attention_layer_name in name:
                    module.register_forward_hook(self.get_attention)

            self.attentions = []

        def get_attention(self, module, input, output):
            self.attentions.append(output.cpu())

        def __call__(self, input_tensor):
            self.attentions = []
            with torch.no_grad():
                output = self.model(input_tensor)
            self.attentions = [self.attentions[-1]]
            return rollout(self.attentions, self.discard_ratio, self.head_fusion)

    def show_mask_on_image(img, mask):
        print(mask.shape)
        img = np.float32(img) / 255
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return np.uint8(255 * cam)

    print("Doing Gradient Attention Rollout")
    vit_attention_grad_rollout = VITAttentionRollout(model, discard_ratio=0.9)
    mask = vit_attention_grad_rollout(input_tensor)
    print(mask.shape)
    name = "grad_rollout_{}_{:.3f}_{}.png".format(0,
                                                  0.9, "max")
    # img = img.to('cpu')
    # np_img = np.array(img)[0, :, :, ::-1]
    # np_img = np_img.transpose(1, 2, 0)
    # print(np_img.shape)

    np_img = np.array(img)[:, :, ::-1]
    np.savez("heatmap.npz", mask=mask, np_img=np_img)

    # features = []
    # labels = []
    # with torch.no_grad():
    #     for data in loader:
    #         inputs, targets = data
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         outputs = model(inputs, only_feat=True, projected=False, simclr=True, ce=True)
    #         features.append(outputs.cpu().numpy())  # 将特征移回 CPU 以便保存
    #         labels.append(targets.cpu().numpy())  # 将标签移回 CPU 以便保存
    #
    # features = np.concatenate(features, axis=0)
    # labels = np.concatenate(labels, axis=0)
    # np.savez('cifar10_features_labels_2.npz', features=features, labels=labels)
