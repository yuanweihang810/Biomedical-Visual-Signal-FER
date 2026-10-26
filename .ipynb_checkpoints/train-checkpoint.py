# paper2/train.py
import os, argparse, random, json
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, average_precision_score
from cale_plus import CALEPlus


class FERDataset(Dataset):
    def __init__(self, df, tf, mode='train'):
        self.df = df.reset_index(drop=True)
        self.tf = tf
        self.mode = mode
        self.emotions = ['angry','disgust','fear','happy','sad','surprise','natural']
        self.emo2idx = {e:i for i,e in enumerate(self.emotions)}

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        if self.tf: img = self.tf(img)
        if self.mode == 'train':
            emo = self.emo2idx[row['emotion']]
            cond = int(row['condition'])
            return img, emo, cond
        else:
            labels = row['labels']  # list of 7 ints
            return img, torch.tensor(labels, dtype=torch.float32)


def train_one_fold(args, train_df, val_df):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CALEPlus(
        backbone_name=args.backbone,
        agg_type=args.agg,
        use_transform=not args.no_transform,
        use_norm=args.use_norm,
    ).to(device)

    train_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    train_ds = FERDataset(train_df, train_tf, 'train')
    val_ds = FERDataset(val_df, val_tf, 'train')
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    bce = nn.BCELoss()

    best_f1, best_state = 0.0, None
    for epoch in range(args.epochs):
        model.train()
        for imgs, emos, conds in train_loader:
            imgs, emos, conds = imgs.to(device), emos.to(device), conds.to(device)
            p, _, _, _ = model(imgs, conds)
            target = F.one_hot(emos, 7).float()
            loss = bce(p, target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()

        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for imgs, emos, conds in val_loader:
                imgs, conds = imgs.to(device), conds.to(device)
                p, _, _, _ = model(imgs, conds)
                preds.append((p > 0.5).float().cpu().numpy())
                gts.append(F.one_hot(emos, 7).float().cpu().numpy())
        preds = np.concatenate(preds, 0); gts = np.concatenate(gts, 0)
        val_f1 = f1_score(gts, preds, average='macro', zero_division=0)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument('--agg', default='A1')
    parser.add_argument('--data_ratio', type=float, default=0.1)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--no_transform', action='store_true')
    parser.add_argument('--use_norm', action='store_true')
    parser.add_argument('--out', default='paper2/exp')
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    df = pd.read_csv('paper2/train_condition_labels.csv')
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_fold = int(len(df) * args.data_ratio)
    fold_size = n_fold // 5
    start = args.fold * fold_size
    end = start + fold_size if args.fold < 4 else n_fold
    train_df = pd.concat([df.iloc[:start], df.iloc[end:n_fold]])
    val_df = df.iloc[start:end]

    os.makedirs(args.out, exist_ok=True)
    model, val_f1 = train_one_fold(args, train_df, val_df)
    tag = f'{args.backbone}_{args.agg}_r{args.data_ratio}_f{args.fold}_s{args.seed}'
    torch.save(model.state_dict(), f'{args.out}/best_{tag}.pth')
    print(f'Saved {tag}, val_macro_f1={val_f1:.4f}')