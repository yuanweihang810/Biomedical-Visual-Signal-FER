# paper2/train.py
import os, argparse, random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, recall_score, accuracy_score
from cale_plus import CALEPlus


EMOTIONS = ['angry','disgust','fear','happy','sad','surprise','natural']
EMO2IDX = {e:i for i,e in enumerate(EMOTIONS)}


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


def train_one_fold(args, train_df, val_df):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CALEPlus(
        backbone_name=args.backbone,
        K=args.K,
        use_transform=not args.no_transform,
        transform_mode=args.transform_mode,
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

    train_loader = DataLoader(FERDataset(train_df, train_tf),
                              batch_size=args.bs, shuffle=True,
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(FERDataset(val_df, val_tf),
                            batch_size=args.bs, shuffle=False,
                            num_workers=8, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1, best_state, best_epoch = 0.0, None, -1
    patience, wait = args.patience, 0

    for epoch in range(args.epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for imgs, emos in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            emos = emos.to(device, non_blocking=True)

            p, logits = model(imgs)
            if torch.isnan(p).any():
                continue

            # 每个分支独立 CE，平均
            K = logits.size(1)
            loss = sum(F.cross_entropy(logits[:, k, :], emos) for k in range(K)) / K

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        # 验证：argmax 单标签口径
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for imgs, emos in val_loader:
                imgs = imgs.to(device)
                p, _ = model(imgs)
                preds.append(p.argmax(dim=1).cpu().numpy())
                gts.append(emos.numpy())
        preds = np.concatenate(preds, 0)
        gts = np.concatenate(gts, 0)
        acc = accuracy_score(gts, preds)
        uar = recall_score(gts, preds, average='macro', zero_division=0)
        val_f1 = f1_score(gts, preds, average='macro', zero_division=0)
        print(f'Epoch {epoch+1}/{args.epochs}: loss={total_loss/max(n_batches,1):.4f}, '
              f'acc={acc:.4f}, uar={uar:.4f}, val_macro_f1={val_f1:.4f}', flush=True)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f'Early stop at epoch {epoch+1}', flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1, best_epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--data_ratio', type=float, default=1.0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--no_transform', action='store_true')
    parser.add_argument('--transform_mode', default='cond',
                        choices=['cond', 'rand'])
    parser.add_argument('--out', default='paper2/exp')
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    df = pd.read_csv('paper2/train_condition_labels.csv')
    df['subject'] = df['path'].apply(lambda p: os.path.basename(p).split('_')[0])

    subjects = df['subject'].unique()
    n_keep = max(1, int(len(subjects) * args.data_ratio))
    rng = np.random.RandomState(args.seed)
    keep = rng.choice(subjects, size=n_keep, replace=False)
    df = df[df['subject'].isin(keep)].reset_index(drop=True)

    gkf = GroupKFold(n_splits=5)
    for i, (tr, va) in enumerate(gkf.split(df, groups=df['subject'])):
        if i == args.fold:
            train_df, val_df = df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)
            break

    print(f'K={args.K} | Train: {len(train_df)} | Val: {len(val_df)} | Subjects: {len(keep)}', flush=True)

    os.makedirs(args.out, exist_ok=True)
    model, best_f1, best_epoch = train_one_fold(args, train_df, val_df)
    if args.no_transform:
        tf_tag = 'notf'
    elif args.transform_mode == 'rand':
        tf_tag = 'rand'
    else:
        tf_tag = 'tf'
    tag = f'{args.backbone}_K{args.K}_{tf_tag}_r{args.data_ratio}_f{args.fold}_s{args.seed}'
    ckpt = f'{args.out}/best_{tag}.pth'
    torch.save(model.state_dict(), ckpt)
    print(f'Saved {ckpt} | best_f1={best_f1:.4f} @ epoch {best_epoch}', flush=True)
