# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from detectron2.layers import nonzero_tuple

__all__ = ["subsample_labels"]


def subsample_labels(
    labels: torch.Tensor, num_samples: int, positive_fraction: float, bg_label: int
):
    """
    Return `num_samples` (or fewer, if not enough found)
    random samples from `labels` which is a mixture of positives & negatives.
    It will try to return as many positives as possible without
    exceeding `positive_fraction * num_samples`, and then try to
    fill the remaining slots with negatives.

    Args:
        labels (Tensor): (N, ) label vector with values:
            * -1: ignore
            * bg_label: background ("negative") class
            * otherwise: one or more foreground ("positive") classes
        num_samples (int): The total number of labels with value >= 0 to return.
            Values that are not sampled will be filled with -1 (ignore).
        positive_fraction (float): The number of subsampled labels with values > 0
            is `min(num_positives, int(positive_fraction * num_samples))`. The number
            of negatives sampled is `min(num_negatives, num_samples - num_positives_sampled)`.
            In order words, if there are not enough positives, the sample is filled with
            negatives. If there are also not enough negatives, then as many elements are
            sampled as is possible.
        bg_label (int): label index of background ("negative") class.

    Returns:
        pos_idx, neg_idx (Tensor):
            1D vector of indices. The total length of both is `num_samples` or fewer.
    """
    positive = nonzero_tuple((labels != -1) & (labels != bg_label))[0]
    negative = nonzero_tuple(labels == bg_label)[0]

    num_pos = int(num_samples * positive_fraction)
    # protect against not enough positive examples
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    # protect against not enough negative examples
    num_neg = min(negative.numel(), num_neg)

    # randomly select positive and negative examples
    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idx = positive[perm1]
    neg_idx = negative[perm2]
    return pos_idx, neg_idx
    
    
    
def subsample_labels_cbs(labels: torch.Tensor, num_samples: int, positive_fraction: float, bg_label: int, classes: torch.Tensor):
    # 找到所有非背景的ground truth类别
    unique_fg_classes = classes[classes != bg_label]

    # 计算每个非背景类别应该采样的正样本数
    num_fg_classes = len(unique_fg_classes)
    per_class_samples = int(num_samples * positive_fraction / num_fg_classes)

    pos_idx_list = []
    neg_idx_list = []

    # 对于每个非背景ground truth类别，采样正样本
    for cls_id in unique_fg_classes:
        # 找到所有标记为该类别的正样本索引
        fg_mask = (classes == cls_id)
        matched_gt_mask = (labels > 0)
        cls_indices = torch.nonzero(fg_mask & matched_gt_mask, as_tuple=False).view(-1)

        # 如果正样本数量不足，需要进行重复采样
        if len(cls_indices) < per_class_samples:
            perm = torch.randperm(len(cls_indices), device=labels.device)
            cls_indices = cls_indices[perm].repeat((per_class_samples + len(cls_indices) - 1) // len(cls_indices))

        # 采样所需数量的正样本
        pos_idx_list.append(cls_indices[:per_class_samples])

    # 对于背景类别，采样负样本
    neg_indices = torch.nonzero(labels == 0, as_tuple=False).view(-1)
    neg_samples = num_samples - len(torch.cat(pos_idx_list))
    if len(neg_indices) < neg_samples:
        neg_indices = neg_indices.repeat((neg_samples + len(neg_indices) - 1) // len(neg_indices))
    neg_idx_list.append(neg_indices[:neg_samples])

    # 合并所有类别的正负样本索引
    pos_idx = torch.cat(pos_idx_list) if pos_idx_list else torch.tensor([], dtype=torch.int64, device=labels.device)
    neg_idx = torch.cat(neg_idx_list) if neg_idx_list else torch.tensor([], dtype=torch.int64, device=labels.device)
    
    # 确保正负样本总数等于 num_samples
    assert len(pos_idx) + len(neg_idx) == num_samples, "The total number of positive and negative samples should be equal to num_samples."

    return pos_idx, neg_idx




