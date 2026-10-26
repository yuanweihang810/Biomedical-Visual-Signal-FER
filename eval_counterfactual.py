# paper2/eval_counterfactual.py
"""W3 反事实变换检验：
对训练好的 K=4 tf 模型，测试时把分支输入的变换"错误重定向"，观察性能退化。
- correct:     分支 k 接收 T_k(x)（正常）
- all_dark:    所有分支接收 T_0(x)
- all_bright:  所有分支接收 T_1(x)
- all_std:     所有分支接收 T_std(x)
- swap:        分支 k 接收 T_{(k+2)%4}(x)
- identity:    所有分支接收原图
"""
import os, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score, recall_score
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


def forward_cf(model, x, mode):
    tf = model.transform
    logits_list = []
    for k in range(model.K):
        if mode == 'correct':
            x_k = tf(x, k)
        elif mode == 'all_dark':
            x_k = tf(x, 0)
        elif mode == 'all_bright':
            x_k = tf(x, 1)
        elif mode == 'all_std':
            x_k = tf(x, 3)
        elif mode == 'swap':
            x_k = tf(x, (k + 2) % 4)
        elif mode == 'identity':
            x_k = x
        else:
            x_k = x
        feat = model.backbone(x_k)
        logits_list.append(model.branch_heads[k](feat))
    logits = torch.stack(logits_list, dim=1)
    return logits.mean(dim=1)


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

    print(f'{"mode":<12} {"acc":>8} {"uar":>8} {"macro_f1":>10}')
    print('-' * 42)
    for mode in ['correct', 'all_dark', 'all_bright', 'all_std', 'swap', 'identity']:
        preds, gts = [], []
        with torch.no_grad():
            for imgs, emos in loader:
                imgs = imgs.to(device)
                p = forward_cf(model, imgs, mode)
                preds.append(p.argmax(dim=1).cpu().numpy())
                gts.append(emos.numpy())
        preds = np.concatenate(preds, 0)
        gts = np.concatenate(gts, 0)
        acc = accuracy_score(gts, preds)
        uar = recall_score(gts, preds, average='macro', zero_division=0)
        f1 = f1_score(gts, preds, average='macro', zero_division=0)
        print(f'{mode:<12} {acc:>8.4f} {uar:>8.4f} {f1:>10.4f}', flush=True)
