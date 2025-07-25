import torch
from torch import nn
from torch.nn import functional as F


def nt_xent_loss(emb_i, emb_j, temperature, gpu):
    outputs = torch.cat((emb_i, emb_j), dim=0)
    B = outputs.shape[0]
    outputs_norm = outputs / (outputs.norm(dim=1).view(B, 1) + 1e-8)
    similarity_matrix = (1. / temperature) * torch.mm(outputs_norm, outputs_norm.transpose(0, 1))
    N2 = len(similarity_matrix)
    N = int(len(similarity_matrix) / 2)

    # Removing diagonal #
    similarity_matrix_exp = torch.exp(similarity_matrix)
    similarity_matrix_exp = similarity_matrix_exp * (1 - torch.eye(N2, N2)).cuda(gpu)

    NT_xent_loss = - torch.log(
        similarity_matrix_exp / (torch.sum(similarity_matrix_exp, dim=1).view(N2, 1) + 1e-8) + 1e-8)
    NT_xent_loss_total = (1. / float(N2)) * torch.sum(
        torch.diag(NT_xent_loss[0:N, N:]) + torch.diag(NT_xent_loss[N:, 0:N]))

    return NT_xent_loss_total


class NTxentLoss(nn.Module):
    def __init__(self, gpu=0, temperature=0.5):
        super().__init__()
        self.gpu = gpu
        self.register_buffer("temperature", torch.tensor(temperature).cuda(gpu))  # 超参数 温度

    def forward(self, emb_i, emb_j):  # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        return nt_xent_loss(emb_i, emb_j, self.temperature, self.gpu)
