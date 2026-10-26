# paper2/cale_plus.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConditionSpecificTransform:
    """mode='cond': 条件特定对比度拉伸；mode='rand': 每分支固定随机几何变换"""
    def __init__(self, lam=20.0, low_res_scale=0.3, mode='cond'):
        self.lam = lam
        self.low_res_scale = low_res_scale
        self.mode = mode

    def _rotate(self, x, degrees):
        import torchvision.transforms.functional as TF
        return TF.rotate(x, degrees)

    def __call__(self, x, c):
        if self.mode == 'rand':
            return self._rotate(x, [0, 90, 180, 270][c])
        if c == 0:
            return torch.log1p(self.lam * x) / math.log(1 + self.lam)
        elif c == 1:
            return 1.0 - torch.log1p(self.lam * (1 - x)) / math.log(1 + self.lam)
        elif c == 2:
            h, w = x.shape[-2:]
            down = F.interpolate(x, scale_factor=self.low_res_scale,
                                 mode='bilinear', align_corners=False)
            up = F.interpolate(down, size=(h, w),
                               mode='bilinear', align_corners=False)
            return up
        else:
            return x


def create_backbone(name='resnet18', pretrained=True):
    W = 'IMAGENET1K_V1' if pretrained else None
    if name == 'resnet18':
        m = models.resnet18(weights=W); fd = m.fc.in_features; m.fc = nn.Identity()
    elif name == 'resnet34':
        m = models.resnet34(weights=W); fd = m.fc.in_features; m.fc = nn.Identity()
    elif name == 'resnet50':
        m = models.resnet50(weights=W); fd = m.fc.in_features; m.fc = nn.Identity()
    elif name == 'resnet101':
        m = models.resnet101(weights=W); fd = m.fc.in_features; m.fc = nn.Identity()
    elif name == 'mobilenetv2':
        m = models.mobilenet_v2(weights=W); fd = m.classifier[-1].in_features; m.classifier = nn.Identity()
    elif name == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=W); fd = m.classifier[-1].in_features; m.classifier = nn.Identity()
    elif name == 'densenet121':
        m = models.densenet121(weights=W); fd = m.classifier.in_features; m.classifier = nn.Identity()
    else:
        raise ValueError(name)
    return m, fd


class CALEPlus(nn.Module):
    """
    K=1: 单分支 baseline
    K>1: 多分支 + 分支独立 BN + 可学习聚合
    """
    def __init__(self, backbone_name='resnet18', K=4, E=7,
                 use_transform=True, transform_mode='cond',
                 lambda_branch=1.0, lambda_agg=1.0):
        super().__init__()
        self.K, self.E = K, E
        self.use_transform = use_transform
        self.transform_mode = transform_mode
        self.lambda_branch = lambda_branch
        self.lambda_agg = lambda_agg

        self.backbone, feat_dim = create_backbone(backbone_name)
        self.feat_dim = feat_dim

        # 每个分支：独立 BN + 独立 head
        self.branch_bns = nn.ModuleList([
            nn.BatchNorm1d(feat_dim) for _ in range(K)])
        self.branch_heads = nn.ModuleList([
            nn.Linear(feat_dim, E) for _ in range(K)])

        # 可学习聚合：从 K*E 维 logits 学出 K 个样本级权重
        if K > 1:
            self.agg_mlp = nn.Sequential(
                nn.Linear(K * E, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, K),
            )
        else:
            self.agg_mlp = None

        self.transform = ConditionSpecificTransform(mode=transform_mode)
        self.register_buffer('imagenet_mean',
                             torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std',
                             torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # x 输入范围 [0,1]
        logits_list = []
        for k in range(self.K):
            x_k = self.transform(x, k) if self.use_transform else x
            x_k = (x_k - self.imagenet_mean) / self.imagenet_std
            feat = self.backbone(x_k)
            feat = self.branch_bns[k](feat)
            logits_list.append(self.branch_heads[k](feat))
        logits = torch.stack(logits_list, dim=1)  # [B, K, E]

        if self.K == 1:
            p = logits.squeeze(1)
        else:
            flat = logits.flatten(1)              # [B, K*E]
            w = F.softmax(self.agg_mlp(flat), dim=1)  # [B, K]
            p = (w.unsqueeze(-1) * logits).sum(dim=1)  # [B, E]

        return p, logits
