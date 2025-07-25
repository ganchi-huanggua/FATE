import torch
import torch.nn as nn

from torch.nn import functional as F


def l2_loss(tensor1, tensor2):
    """
    计算两组tensor之间的L2损失（欧氏距离）。
    :param tensor1: 第一组tensor，形状为 (batch_size, feature_dim)
    :param tensor2: 第二组tensor，形状为 (batch_size, feature_dim)，应该与tensor1具有相同的形状
    :return: L2损失，形状为 (batch_size,)
    """
    # 计算tensor1和tensor2之间的差的平方
    diff = tensor1 - tensor2
    # 计算差的平方的和（即L2距离的平方）
    l2_squared = torch.sum(diff ** 2, dim=1)
    # 如果需要L2损失（即欧氏距离），则取平方根
    l2_loss = torch.sqrt(l2_squared)
    return l2_loss


class L2SimilarityLoss(nn.Module):
    """
    Wrapper for ce loss
    """
    def forward(self, tensor1, tensor2):
        return l2_loss(tensor1, tensor2)

