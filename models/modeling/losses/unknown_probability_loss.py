import torch
import torch.distributions as dists
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class UPLoss(nn.Module):
    """
    Unknown Probability Loss - 用于处理包含未知类别的分类任务
    """

    def __init__(
        self,
        num_classes: int,
        sampling_metric: str = "max_entropy",
        topk: int = 3,
        alpha: float = 1.0
    ):
        """
        初始化UPLoss
        
        Args:
            num_classes: 类别数量
            sampling_metric: 采样方法，可选["max_entropy", "random"]
            topk: 每类采样的样本数
            alpha: 目标调整参数
        """
        super(UPLoss, self).__init__()
        self.num_classes = num_classes
        
        valid_metrics = ["max_entropy", "random"]
        if sampling_metric not in valid_metrics:
            raise ValueError(f"sampling_metric必须是以下之一: {valid_metrics}")
            
        self.sampling_metric = sampling_metric
        self.topk = topk  # 如果topk=-1，则采样前景样本数量的两倍
        self.alpha = alpha

    def compute_soft_cross_entropy(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """计算软交叉熵损失"""
        log_probabilities = F.log_softmax(predictions, dim=1)
        batch_loss = -(targets * log_probabilities).sum() / predictions.shape[0]
        return batch_loss

    def select_samples(self, scores: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        根据采样策略选择前景和背景样本
        
        Args:
            scores: 预测分数
            labels: 真实标签
            
        Returns:
            前景分数、背景分数、前景标签、背景标签
        """
        # 分离前景和背景样本
        foreground_mask = labels != self.num_classes
        fg_scores, fg_labels = scores[foreground_mask], labels[foreground_mask]
        bg_scores, bg_labels = scores[~foreground_mask], labels[~foreground_mask]

        # 处理分数向量，移除未知类别
        processed_fg_scores = torch.cat(
            [fg_scores[:, :self.num_classes-1], fg_scores[:, -1:]], dim=1)
        processed_bg_scores = torch.cat(
            [bg_scores[:, :self.num_classes-1], bg_scores[:, -1:]], dim=1)

        # 确定采样数量
        fg_count = fg_scores.size(0)
        sample_count = fg_count if (self.topk == -1 or fg_count < self.topk) else self.topk
        
        # 根据不同的采样策略计算指标
        if self.sampling_metric == "max_entropy":
            # 使用最大熵作为不确定性度量
            fg_metric = dists.Categorical(processed_fg_scores.softmax(dim=1)).entropy()
            bg_metric = dists.Categorical(processed_bg_scores.softmax(dim=1)).entropy()
        else:  # "random"
            # 随机选择样本
            fg_metric = torch.rand(processed_fg_scores.size(0)).to(scores.device)
            bg_metric = torch.rand(processed_bg_scores.size(0)).to(scores.device)

        # 选择topk样本
        _, fg_indices = fg_metric.topk(sample_count)
        _, bg_indices = bg_metric.topk(sample_count)
        
        return (
            fg_scores[fg_indices], 
            bg_scores[bg_indices], 
            fg_labels[fg_indices], 
            bg_labels[bg_indices]
        )

    def forward(self, scores: Tensor, labels: Tensor) -> Tensor:
        """
        计算UPLoss
        
        Args:
            scores: 预测分数 [batch_size, num_classes]
            labels: 真实标签 [batch_size]
            
        Returns:
            计算得到的损失值
        """
        # 采样前景和背景样本
        fg_scores, bg_scores, fg_labels, bg_labels = self.select_samples(scores, labels)
        
        # 合并前景和背景样本
        combined_scores = torch.cat([fg_scores, bg_scores])
        combined_labels = torch.cat([fg_labels, bg_labels])

        # 创建掩码以排除目标类别
        batch_size, num_classes = combined_scores.shape
        class_indices = torch.arange(num_classes).repeat(batch_size, 1).to(scores.device)
        exclusion_mask = class_indices != combined_labels[:, None].repeat(1, num_classes)
        masked_indices = class_indices[exclusion_mask].reshape(batch_size, num_classes-1)

        # 获取目标分数和掩码分数
        ground_truth_probs = torch.gather(
            F.softmax(combined_scores, dim=1), 1, combined_labels[:, None]
        ).squeeze(1)
        masked_scores = torch.gather(combined_scores, 1, masked_indices)

        # 确保分数非负
        ground_truth_probs = torch.clamp(ground_truth_probs, min=0.0)
        
        # 创建目标
        targets = torch.zeros_like(masked_scores)
        fg_count = fg_scores.size(0)
        
        # 前景样本目标
        uncertainty_weight_fg = ground_truth_probs[:fg_count] * (1-ground_truth_probs[:fg_count]).pow(self.alpha)
        targets[:fg_count, self.num_classes-2] = uncertainty_weight_fg
        
        # 背景样本目标
        uncertainty_weight_bg = ground_truth_probs[fg_count:] * (1-ground_truth_probs[fg_count:]).pow(self.alpha)
        targets[fg_count:, self.num_classes-1] = uncertainty_weight_bg

        # 计算软交叉熵损失
        return self.compute_soft_cross_entropy(masked_scores, targets.detach())