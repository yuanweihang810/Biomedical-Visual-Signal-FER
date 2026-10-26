# paper2/eval_branch_consistency.py
"""W5 分支一致性：计算 K 个分支在同一验证集上预测的 Cohen's kappa 和各自的 macro-F1。"""
import os, argparse, itertools
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import GroupKFold
from cale_plus import CALEPlus


EMO2IDX = {'angry':0,'disgust':1,'fear':2,'happy':3,
           'sad':4,'surprise':5,'natural':6}


class FERDataset(Dataset):
    def __init__(self, df, tf):
        self.df = df.reset_index(drop=True)
        self.tf = tf
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        if self.tf:
            img = self.tf(img)
        return img, EMO2IDX[row['emotion']]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv('paper2/train_condition_labels.csv')
    df['subject'] = df['path'].apply(lambda p: os.path.basename(p).split('_')[0])
    gkf = GroupKFold(n_splits=5)
    for i, (tr, va) in enumerate(gkf.split(df, groups=df['subject'])):
        if i == args.fold:
            val_df = df.iloc[va].reset_index(drop=True)
            break

    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    loader = DataLoader(FERDataset(val_df, tf), batch_size=64,
                        shuffle=False, num_workers=8)

    model = CALEPlus(backbone_name=args.backbone, K=args.K).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    branch_preds = {k: [] for k in range(model.K)}
    gts = []
    with torch.no_grad():
        for imgs, emos in loader:
            imgs = imgs.to(device)
            for k in range(model.K):
                x_k = model.transform(imgs, k)
                feat = model.backbone(x_k)
                logit = model.branch_heads[k](feat)
                branch_preds[k].append(logit.argmax(dim=1).cpu().numpy())
            gts.append(emos.numpy())
    gts = np.concatenate(gts, 0)
    for k in range(model.K):
        branch_preds[k] = np.concatenate(branch_preds[k], 0)

    print('=== Each branch vs ground truth ===')
    for k in range(model.K):
        f1 = f1_score(gts, branch_preds[k], average='macro', zero_division=0)
        print(f'branch {k}: val_macro_f1 = {f1:.4f}')

    print('\n=== Pairwise Cohen kappa (branch-branch agreement) ===')
    for i, j in itertools.combinations(range(model.K), 2):
        kappa = cohen_kappa_score(branch_preds[i], branch_preds[j])
        print(f'branch {i} vs branch {j}: kappa = {kappa:.4f}')

    print('\n=== Branch-condition cross matrix (mAP-like F1) ===')
    cond = val_df['condition'].values
    K = model.K
    M = np.zeros((K, K))  # [cond, branch]
    for c in range(K):
        mask = (cond == c)
        if mask.sum() == 0:
            continue
        for k in range(K):
            M[c, k] = f1_score(gts[mask], branch_preds[k][mask],
                               average='macro', zero_division=0)
    print('rows: condition c, cols: branch k, value: macro-F1')
    print('       ' + ' '.join(f'k={k:<6}' for k in range(K)))
    for c in range(K):
        print(f'c={c}:  ' + ' '.join(f'{M[c,k]:.4f}' for k in range(K)))
