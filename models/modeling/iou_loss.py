import torch
import math
import torch.nn.functional as F

def gbb_form(boxes):
    xy, wh, angle = torch.split(boxes, [2, 2, 1], dim=-1)
    angle = angle * math.pi / 180
    return torch.cat([xy, wh.pow(2) / 12., angle], dim=-1)


def rotated_form(a_, b_, angles):
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    a = a_ * torch.pow(cos_a, 2) + b_ * torch.pow(sin_a, 2)
    b = a_ * torch.pow(sin_a, 2) + b_ * torch.pow(cos_a, 2)
    c = (a_ - b_) * cos_a * sin_a
    return a, b, c


def probiou_loss(pred, target, eps=1e-3, mode='l1'):
    """
        pred    -> a matrix [N,5](x,y,w,h,angle - in radians) containing ours predicted box ;in case of HBB angle == 0
        target  -> a matrix [N,5](x,y,w,h,angle - in radians) containing ours target    box ;in case of HBB angle == 0
        eps     -> threshold to avoid infinite values
        mode    -> ('l1' in [0,1] or 'l2' in [0,inf]) metrics according our paper

    """
    
    gbboxes1 = gbb_form(pred)
    gbboxes2 = gbb_form(target)

    x1, y1, a1_, b1_, c1_ = gbboxes1[:,
                                     0], gbboxes1[:,
                                                  1], gbboxes1[:,
                                                               2], gbboxes1[:,
                                                                            3], gbboxes1[:,
                                                                                         4]
    x2, y2, a2_, b2_, c2_ = gbboxes2[:,
                                     0], gbboxes2[:,
                                                  1], gbboxes2[:,
                                                               2], gbboxes2[:,
                                                                            3], gbboxes2[:,
                                                                                         4]

    a1, b1, c1 = rotated_form(a1_, b1_, c1_)
    a2, b2, c2 = rotated_form(a2_, b2_, c2_)

    t1 = 0.25 * ((a1 + a2) * (torch.pow(y1 - y2, 2)) + (b1 + b2) * (torch.pow(x1 - x2, 2))) + \
         0.5 * ((c1+c2)*(x2-x1)*(y1-y2))
    t2 = (a1 + a2) * (b1 + b2) - torch.pow(c1 + c2, 2)
    t3_ = (a1 * b1 - c1 * c1) * (a2 * b2 - c2 * c2)
    t3 = 0.5 * torch.log(t2 / (4 * torch.sqrt(F.relu(t3_)) + eps))

    B_d = (t1 / t2) + t3
    # B_d = t1 + t2 + t3

    B_d = torch.clip(B_d, min=eps, max=100.0)
    l1 = torch.sqrt(1.0 - torch.exp(-B_d) + eps)
    l_i = torch.pow(l1, 2.0)
    l2 = -torch.log(1.0 - l_i + eps)

    if mode == 'l1':
        probiou = l1
    if mode == 'l2':
        probiou = l2
        
    
    probiou = probiou.mean() if probiou.numel() > 0 else 0.0 * probiou.sum()
    return probiou