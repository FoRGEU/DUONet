import torch
import torch.nn as nn
import torch.nn.functional as F


class ICLoss(nn.Module):
    """ Instance Contrastive Loss
    """
    def __init__(self, temperature=0.1):
        super(ICLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels, queue_features, queue_labels):
        # 将特征移动到相同设备
        device = features.device
        
        # 创建相似标签掩码
        similarity_matrix = (labels.view(-1, 1) == queue_labels.view(1, -1)).float().to(device)
        
        # 计算余弦相似度并应用温度缩放
        similarity_scores = torch.mm(features, queue_features.T) / self.temperature
        
        # 为数值稳定性调整logits
        similarity_scores = similarity_scores - similarity_scores.max(dim=1, keepdim=True)[0].detach()
        
        # 创建有效logits掩码(排除自身比较)
        valid_mask = torch.ones_like(similarity_scores, device=device)
        valid_mask[similarity_scores == 0] = 0
        
        # 应用掩码
        effective_mask = similarity_matrix * valid_mask
        
        # 计算softmax概率
        exp_sim = torch.exp(similarity_scores) * valid_mask
        log_probs = similarity_scores - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
        
        # 计算正样本的平均对数似然
        # 避免除零
        positive_counts = effective_mask.sum(dim=1)
        positive_counts = torch.clamp(positive_counts, min=1e-8)
        positive_log_probs = (effective_mask * log_probs).sum(dim=1) / positive_counts
        
        # 计算最终损失
        loss = -positive_log_probs.mean()
        
        # 防止NaN值
        return loss if torch.isfinite(loss) else torch.tensor(0.0, device=device)