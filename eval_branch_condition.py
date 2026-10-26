# paper2/eval_branch_condition.py
"""分支-条件交叉矩阵：每个条件的样本，用每个分支单独预测的准确率。
   对角线高说明分支条件专业化成立。"""
import os, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score
from cale_plus import CALEPlus

EMO2IDX = {'angry':0,'disgust':1,'fear':2,'happy':3,
           'sad':4,'surprise':5,'natural':6}

class DS(Dataset):
    def __init__(self, df, tf):
        self.df = df.reset_index(drop=True); self.tf = tf
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(r['path']).convert('RGB')
        if self.tf: img = self.tf(img)
        return img, EMO2IDX[r['emotion']], int(r['condition'])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--backbone', default='resnet18')
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--fold', type=int, default=0)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('paper2/train_condition_labels.csv')
    df['subject'] = df['path'].apply(lambda p: os.path.basename(p).split('_')[0])
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)
    for i, (tr, va) in enumerate(gkf.split(df, groups=df['subject'])):
        if i == args.fold:
            val_df = df.iloc[va].reset_index(drop=True); break

    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    loader = DataLoader(DS(val_df, tf), batch_size=64, shuffle=False, num_workers=8)

    model = CALEPlus(backbone_name=args.backbone, K=args.K).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # 收集所有分支的预测
    K = model.K
    per_branch_preds = {k: [] for k in range(K)}
    gts, conds = [], []
    with torch.no_grad():
        for imgs, emos, c in loader:
            imgs = imgs.to(device)
            for k in range(K):
                x_k = model.transform(imgs, k)
                x_k = (x_k - model.imagenet_mean) / model.imagenet_std
                feat = model.backbone(x_k)
                feat = model.branch_bns[k](feat)
                logits = model.branch_heads[k](feat)
                per_branch_preds[k].append(logits.argmax(1).cpu().numpy())
            gts.append(emos.numpy()); conds.append(c.numpy())
    gts = np.concatenate(gts); conds = np.concatenate(conds)
    for k in range(K):
        per_branch_preds[k] = np.concatenate(per_branch_preds[k])

    print('=== Per-branch overall accuracy ===')
    for k in range(K):
        print(f'  branch {k}: {accuracy_score(gts, per_branch_preds[k]):.4f}')

    print('\n=== Branch-condition cross matrix (rows=condition, cols=branch) ===')
    print('        ' + '  '.join(f'k={k}' for k in range(K)))
    for c in range(K):
        row = []
        for k in range(K):
            mask = (conds == c)
            row.append(accuracy_score(gts[mask], per_branch_preds[k][mask]))
        print(f'  c={c}:  ' + '  '.join(f'{v:.4f}' for v in row))

    # 平均聚合后的准确率
    avg_pred = np.mean([per_branch_preds[k] for k in range(K)], axis=0)
    # 注意：这里只做示意，实际平均聚合是在 logits 空间做
    print(f'\nAcc (majority vote): {accuracy_score(gts, avg_pred):.4f}')
