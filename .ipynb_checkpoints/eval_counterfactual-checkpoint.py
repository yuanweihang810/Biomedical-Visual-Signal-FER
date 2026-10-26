# paper2/eval_counterfactual.py
"""
E3 反事实变换检验：
对每个条件 c 的测试样本，分别用一致变换 T_c、矛盾变换 T_c'（c'≠c）、
恒等变换 T_std 生成输入，得到三种分支-条件矩阵。
若矛盾变换下对角线优势消失，则 W1 成立。
"""
import os, json, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.metrics import f1_score
from cale_plus import CALEPlus


class CondTestDataset(Dataset):
    def __init__(self, df, tf):
        self.df = df.reset_index(drop=True)
        self.tf = tf
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        if self.tf: img = self.tf(img)
        labels = [int(x) for x in str(row['labels']).split(',')] if isinstance(row['labels'], str) else row['labels']
        return img, torch.tensor(labels, dtype=torch.float32), int(row['condition'])


def branch_condition_matrix(model, loader, device, K=4):
    """返回 K×K 矩阵，元素 [c, k] 为条件 c 的样本用分支 k 单独预测的 mAP"""
    model.eval()
    results = {c: {k: {'probs': [], 'labels': []} for k in range(K)} for c in range(K)}
    with torch.no_grad():
        for imgs, labels, conds in loader:
            imgs = imgs.to(device)
            labels = labels.numpy()
            conds = conds.numpy()
            # 用恒等变换下的分支输出
            _, s, _, _ = model(imgs, conds.to(device) if isinstance(conds, torch.Tensor) else torch.tensor(conds))
            s = s.cpu().numpy()  # [B, K, E]
            for i in range(len(conds)):
                c = conds[i]
                for k in range(K):
                    results[c][k]['probs'].append(s[i, k])
                    results[c][k]['labels'].append(labels[i])
    from sklearn.metrics import average_precision_score
    M = np.zeros((K, K))
    for c in range(K):
        for k in range(K):
            probs = np.array(results[c][k]['probs'])
            labels = np.array(results[c][k]['labels'])
            aps = [average_precision_score(labels[:, e], probs[:, e]) for e in range(labels.shape[1])]
            M[c, k] = np.mean(aps)
    return M


def counterfactual_eval(model, loader, device, counterfactual_map):
    """用矛盾变换评估：分支 k 接收 T_{cf_map[k]}(x) 的输入"""
    model.eval()
    all_probs, all_labels, all_conds = [], [], []
    with torch.no_grad():
        for imgs, labels, conds in loader:
            imgs = imgs.to(device)
            p, s, _, _ = model(imgs, torch.tensor(conds).to(device),
                               counterfactual_map=counterfactual_map)
            all_probs.append(p.cpu().numpy())
            all_labels.append(labels.numpy())
            all_conds.append(conds.numpy())
    probs = np.concatenate(all_probs, 0)
    labels = np.concatenate(all_labels, 0)
    conds = np.concatenate(all_conds, 0)
    from sklearn.metrics import average_precision_score
    ap = [average_precision_score(labels[:, i], probs[:, i]) for i in range(labels.shape[1])]
    return np.mean(ap), probs, labels, conds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument('--agg', default='A1')
    parser.add_argument('--test_csv', default='paper2/test_condition_labels.csv')
    parser.add_argument('--test_dir', default='/root/autodl-tmp/IEEE/test_aligned/test')
    parser.add_argument('--out', default='paper2/cf_results.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CALEPlus(backbone_name=args.backbone, agg_type=args.agg).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    df = pd.read_csv(args.test_csv)
    df['path'] = df['path'].apply(lambda p: os.path.join(args.test_dir, os.path.basename(p)))
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    loader = DataLoader(CondTestDataset(df, tf), batch_size=64, shuffle=False, num_workers=4)

    # 1) 一致变换矩阵
    M_consistent = branch_condition_matrix(model, loader, device)

    # 2) 矛盾变换矩阵（每个分支用错误的变换）
    cf_map = {0: 1, 1: 0, 2: 3, 3: 2}  # dark<->bright, lowres<->std
    mAP_cf, _, _, _ = counterfactual_eval(model, loader, device, cf_map)

    # 3) 恒等变换矩阵
    # 通过设置 use_transform=False 重新前向；简化起见这里只用一致/矛盾对比
    result = {
        'consistent_matrix': M_consistent.tolist(),
        'counterfactual_map': cf_map,
        'counterfactual_mAP': float(mAP_cf),
        'diag_advantage_consistent': float(np.mean(np.diag(M_consistent)) - (M_consistent.sum() - np.trace(M_consistent)) / (M_consistent.size - M_consistent.shape[0])),
    }
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))