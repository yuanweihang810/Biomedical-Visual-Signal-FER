# paper2/cale_plus.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================
# 条件特定变换 T_c（与合成条件生成解耦：变换函数不同）
# ============================================================
class ConditionSpecificTransform:
    """
    c=0 dark:    log transform, stretch dark range
    c=1 bright:  gamma correction, compress highlights
    c=2 lowres:  downsample + upsample
    c=3 std:     identity
    注意：这里的变换函数与合成数据生成时使用的变换不同，
    以便在反事实检验中区分"条件感知"与"变换识别"。
    """
    def __init__(self, low_res_scale=0.3, log_lambda=20.0, gamma=1.6):
        self.low_res_scale = low_res_scale
        self.log_lambda = log_lambda
        self.gamma = gamma

    def __call__(self, x, c):
        if c == 0:
            return torch.log1p(self.log_lambda * x) / math.log(1 + self.log_lambda)
        elif c == 1:
            return x ** self.gamma
        elif c == 2:
            h, w = x.shape[-2:]
            down = F.interpolate(x, scale_factor=self.low_res_scale,
                                 mode='bilinear', align_corners=False)
            up = F.interpolate(down, size=(h, w),
                               mode='bilinear', align_corners=False)
            return up
        else:
            return x


# ============================================================
# 共享 backbone
# ============================================================
def create_backbone(name='resnet18', pretrained=True):
    if name == 'resnet18':
        m = models.resnet18(pretrained=pretrained)
        fd = m.fc.in_features; m.fc = nn.Identity()
    elif name == 'resnet34':
        m = models.resnet34(pretrained=pretrained)
        fd = m.fc.in_features; m.fc = nn.Identity()
    elif name == 'resnet50':
        m = models.resnet50(pretrained=pretrained)
        fd = m.fc.in_features; m.fc = nn.Identity()
    elif name == 'mobilenetv2':
        m = models.mobilenet_v2(pretrained=pretrained)
        fd = m.classifier[-1].in_features; m.classifier = nn.Identity()
    elif name == 'efficientnet_b0':
        m = models.efficientnet_b0(pretrained=pretrained)
        fd = m.classifier[-1].in_features; m.classifier = nn.Identity()
    else:
        raise ValueError(name)
    return m, fd


# ============================================================
# 五种聚合机制 A0-A4
# ============================================================
class AggregationA0(nn.Module):
    """固定求和：p_e = s_{e,c*} + alpha * sum_{k != c*} s_{e,k}"""
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, s, cond_onehot):
        # s: [B, K, E], cond_onehot: [B, K]
        main = (s * cond_onehot.unsqueeze(-1)).sum(dim=1)  # [B, E]
        non_main = s.sum(dim=1) - main
        return main + self.alpha * non_main


class AggregationA1(nn.Module):
    """可学习标量权重：p_e = sum_k w_k * s_{e,k}"""
    def __init__(self, K=4, init=None):
        super().__init__()
        if init is None:
            init = [1.0] + [0.5] * (K - 1)
        self.w = nn.Parameter(torch.tensor(init))

    def forward(self, s, cond_onehot):
        w = self.w.unsqueeze(0).unsqueeze(-1)  # [1, K, 1]
        return (w * s).sum(dim=1)


class AggregationA2(nn.Module):
    """条件条件化权重：每个条件 c 对应一组权重 w_k(c)"""
    def __init__(self, K=4, init=None):
        super().__init__()
        if init is None:
            init = [[1.0 if k == c else 0.5 for k in range(K)] for c in range(K)]
        self.w = nn.Parameter(torch.tensor(init, dtype=torch.float32))  # [K, K]

    def forward(self, s, cond_onehot):
        # w_c: [B, K] 每个样本根据其条件 c 取 w[c]
        w = torch.matmul(cond_onehot, self.w)  # [B, K]
        return (w.unsqueeze(-1) * s).sum(dim=1)


class AggregationA3(nn.Module):
    """注意力聚合：alpha_{e,k} = softmax_k(MLP([feat_k, s_{e,k}]))"""
    def __init__(self, K=4, E=7, feat_dim=512):
        super().__init__()
        self.K, self.E = K, E
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + 1, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, s, cond_onehot, feats=None):
        # s: [B, K, E], feats: [B, K, feat_dim]
        B, K, E = s.shape
        # 展开为 [B*K*E, feat_dim+1]
        feats_rep = feats.unsqueeze(2).expand(-1, -1, E, -1).reshape(B*K*E, -1)
        s_flat = s.reshape(B*K*E, 1)
        alpha = self.mlp(torch.cat([feats_rep, s_flat], dim=-1))  # [B*K*E, 1]
        alpha = alpha.reshape(B, K, E)
        alpha = F.softmax(alpha, dim=1)
        return (alpha * s).sum(dim=1)


class AggregationA4(nn.Module):
    """归一化聚合：p_e = sum_k w_k s_{e,k} / sum_k w_k"""
    def __init__(self, K=4, init=None):
        super().__init__()
        if init is None:
            init = [1.0] + [0.5] * (K - 1)
        self.w = nn.Parameter(torch.tensor(init))

    def forward(self, s, cond_onehot):
        w = self.w.unsqueeze(0).unsqueeze(-1)
        num = (w * s).sum(dim=1)
        den = self.w.sum()
        return num / (den + 1e-8)


# ============================================================
# CALE++ 主模型
# ============================================================
class CALEPlus(nn.Module):
    def __init__(self, backbone_name='resnet18', K=4, E=7,
                 agg_type='A1', use_transform=True, use_norm=False):
        super().__init__()
        self.K, self.E = K, E
        self.agg_type = agg_type
        self.use_transform = use_transform
        self.backbone, feat_dim = create_backbone(backbone_name)
        self.feat_dim = feat_dim
        # 每个条件一个分支头
        self.branch_heads = nn.ModuleList([
            nn.Linear(feat_dim, E) for _ in range(K)
        ])
        self.transform = ConditionSpecificTransform()

        # 聚合模块
        if agg_type == 'A0':
            self.agg = AggregationA0()
        elif agg_type == 'A1':
            self.agg = AggregationA1(K)
        elif agg_type == 'A2':
            self.agg = AggregationA2(K)
        elif agg_type == 'A3':
            self.agg = AggregationA3(K, E, feat_dim)
        elif agg_type == 'A4':
            self.agg = AggregationA4(K)
        else:
            raise ValueError(agg_type)

        self.use_norm = use_norm

    def _forward_branches(self, x, cond_indices):
        """对每个条件分支生成变换输入并前向，返回 s 和 feats"""
        B = x.shape[0]
        s_list, feat_list = [], []
        for k in range(self.K):
            x_k = self.transform(x, k) if self.use_transform else x
            feat = self.backbone(x_k)
            logit = self.branch_heads[k](feat)
            s = torch.sigmoid(logit)
            s_list.append(s)
            feat_list.append(feat)
        s = torch.stack(s_list, dim=1)         # [B, K, E]
        feats = torch.stack(feat_list, dim=1)  # [B, K, feat_dim]
        return s, feats

    def forward(self, x, cond_indices, agg_input='gt',
                counterfactual_map=None):
        """
        x: [B, 3, H, W]
        cond_indices: [B] ground-truth 条件
        agg_input: 'gt' | 'pred' | 'random'
        counterfactual_map: dict {cond: alt_cond} 用于反事实变换检验
        """
        B = x.shape[0]
        device = x.device

        # 构造 one-hot 条件
        if agg_input == 'random':
            cond_used = torch.randint(0, self.K, (B,), device=device)
        else:
            cond_used = cond_indices
        cond_oh = F.one_hot(cond_used, num_classes=self.K).float()

        # 分支前向
        s, feats = self._forward_branches(x, cond_used)

        # 反事实：若指定了 counterfactual_map，则对输入施加错误变换
        if counterfactual_map is not None:
            s_cf_list = []
            for k in range(self.K):
                # 该分支本应使用的条件映射到另一个条件
                alt_k = counterfactual_map.get(k, k)
                x_cf = self.transform(x, alt_k)
                feat_cf = self.backbone(x_cf)
                s_cf = torch.sigmoid(self.branch_heads[k](feat_cf))
                s_cf_list.append(s_cf)
            s = torch.stack(s_cf_list, dim=1)

        # 聚合
        if self.agg_type == 'A3':
            p = self.agg(s, cond_oh, feats)
        else:
            p = self.agg(s, cond_oh)

        # 归一化（可选）
        if self.use_norm and self.agg_type != 'A4':
            p = p / (p.max(dim=1, keepdim=True)[0] + 1e-8)

        return p, s, cond_oh, feats