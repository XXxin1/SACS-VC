import torch
from torch import nn


class PatchNCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.nce_T = 0.09

    def forward(self, feat_q, feat_k):
        B, C, S = feat_q.shape
        feat_k = feat_k.detach()
        l_pos = (feat_q * feat_k).sum(dim=1)[:, :, None]
        feat_q = feat_q.permute(0, 2, 1)
        l_neg = torch.bmm(feat_q, feat_k)
        diagonal = torch.eye(S, device=feat_q.device, dtype=torch.bool)
        l_neg.masked_fill_(diagonal, -10.0)
        logits = torch.cat((l_pos, l_neg), dim=2) / self.nce_T
        predictions = logits.flatten(0, 1)
        targets = torch.zeros(B * S, device=feat_q.device, dtype=torch.long)
        loss = self.cross_entropy_loss(predictions, targets)
        return loss