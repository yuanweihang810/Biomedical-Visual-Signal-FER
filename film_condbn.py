# paper2/film_condbn.py
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def create_backbone(name='resnet18', pretrained=True):
    W = 'IMAGENET1K_V1' if pretrained else None
    if name == 'resnet18':
        m = models.resnet18(weights=W); fd = m.fc.in_features; m.fc = nn.Identity()
    elif name == 'resnet50':
        m = models.resnet50(weights=W); fd = m.fc.in_features; m.fc = nn.Identity()
    return m, fd


class ConditionTransform:
    def __init__(self, lam=20.0, low_res_scale=0.3):
        self.lam = lam; self.low_res_scale = low_res_scale
    def __call__(self, x, c):
        if c == 0:
            return torch.log1p(self.lam * x) / math.log(1 + self.lam)
        elif c == 1:
            return 1.0 - torch.log1p(self.lam * (1 - x)) / math.log(1 + self.lam)
        elif c == 2:
            h, w = x.shape[-2:]
            down = F.interpolate(x, scale_factor=self.low_res_scale,
                                 mode='bilinear', align_corners=False)
            return F.interpolate(down, size=(h, w), mode='bilinear', align_corners=False)
        return x


class FiLM(nn.Module):
    def __init__(self, feat_dim, num_conditions=4):
        super().__init__()
        self.emb = nn.Embedding(num_conditions, feat_dim * 2)
        nn.init.zeros_(self.emb.weight)
    def forward(self, feat, cond_idx):
        gb = self.emb(cond_idx)
        g, b = gb.chunk(2, dim=1)
        return (1 + g) * feat + b


class CondBN(nn.Module):
    def __init__(self, feat_dim, num_conditions=4):
        super().__init__()
        self.bns = nn.ModuleList([nn.BatchNorm1d(feat_dim) for _ in range(num_conditions)])
    def forward(self, feat, cond_idx):
        out = torch.zeros_like(feat)
        for c in range(len(self.bns)):
            m = (cond_idx == c)
            n = m.sum().item()
            if n > 1:
                out[m] = self.bns[c](feat[m])
            elif n == 1:
                # 单样本时 BN 在 train 模式不可用，退化为恒等
                out[m] = feat[m]
        return out


class FiLM_CALE(nn.Module):
    def __init__(self, backbone_name='resnet18', num_emotions=7, num_conditions=4):
        super().__init__()
        self.backbone, fd = create_backbone(backbone_name)
        self.film = FiLM(fd, num_conditions)
        self.head = nn.Linear(fd, num_emotions)
        self.transform = ConditionTransform()
        self.register_buffer('mean', torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))
    def forward(self, x, cond_idx):
        x_t = self.transform(x, 3)  # identity，因为 FiLM 在特征层调制
        x_t = (x_t - self.mean) / self.std
        feat = self.backbone(x_t)
        feat = self.film(feat, cond_idx)
        return self.head(feat)


class CondBN_CALE(nn.Module):
    def __init__(self, backbone_name='resnet18', num_emotions=7, num_conditions=4):
        super().__init__()
        self.backbone, fd = create_backbone(backbone_name)
        self.condbn = CondBN(fd, num_conditions)
        self.head = nn.Linear(fd, num_emotions)
        self.transform = ConditionTransform()
        self.register_buffer('mean', torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))
    def forward(self, x, cond_idx):
        x_t = self.transform(x, 3)
        x_t = (x_t - self.mean) / self.std
        feat = self.backbone(x_t)
        feat = self.condbn(feat, cond_idx)
        return self.head(feat)
