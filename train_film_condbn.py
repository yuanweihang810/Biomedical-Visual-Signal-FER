# paper2/train_film_condbn.py
"""FiLM / CondBN 基线对比。条件标签使用 3 类版本（暗光/强光/标准）。"""
import os, argparse, random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
from film_condbn import FiLM_CALE, CondBN_CALE
from rafdb_dataset import load_rafdb

EMO2IDX = {'angry':0,'disgust':1,'fear':2,'happy':3,
           'sad':4,'surprise':5,'natural':6}


class RAFDBWithCond(Dataset):
    def __init__(self, items, cond_map, img_dir, tf):
        # items: list of (filename, label_our_idx) from load_rafdb
        self.items = items
        self.cond_map = cond_map  # filename -> condition
        self.img_dir = img_dir
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fn, lab = self.items[idx]
        img = Image.open(os.path.join(self.img_dir, fn)).convert('RGB')
        if self.tf:
            img = self.tf(img)
        cond = self.cond_map.get(fn, 2)  # default to standard
        return img, lab, cond


def build_cond_map(csv_path):
    df = pd.read_csv(csv_path)
    # 把 condition=3（standard）映射为 2，因为 num_conditions=3
    df['condition'] = df['condition'].map({0: 0, 1: 1, 3: 2})
    return dict(zip(df['filename'], df['condition']))


def evaluate(model, loader, device, is_film):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for imgs, labs, conds in loader:
            imgs = imgs.to(device)
            conds = conds.to(device)
            logits = model(imgs, conds)
            preds.append(logits.argmax(1).cpu().numpy())
            gts.append(labs.numpy())
    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    return (accuracy_score(gts, preds),
            recall_score(gts, preds, average='macro', zero_division=0),
            f1_score(gts, preds, average='macro', zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['film', 'condbn'], required=True)
    ap.add_argument('--backbone', default='resnet18')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--out', default='paper2/rafdb_film_condbn')
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 载入数据
    train_all, test_items, img_dir = load_rafdb('/root/autodl-tmp/IEEE')
    cond_map = build_cond_map('/root/autodl-tmp/IEEE/paper2/rafdb_condition_labels_3class.csv')
    print(f'RAF-DB: {len(train_all)} train | {len(test_items)} test')
    print(f'cond_map: {len(cond_map)} entries')

    labels = [lab for _, lab in train_all]
    train_items, val_items = train_test_split(
        train_all, test_size=args.val_ratio,
        random_state=args.seed, stratify=labels)

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(RAFDBWithCond(train_items, cond_map, img_dir, train_tf),
                              batch_size=args.bs, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(RAFDBWithCond(val_items, cond_map, img_dir, eval_tf),
                            batch_size=args.bs, shuffle=False,
                            num_workers=8, pin_memory=True)
    test_loader = DataLoader(RAFDBWithCond(test_items, cond_map, img_dir, eval_tf),
                             batch_size=args.bs, shuffle=False,
                             num_workers=8, pin_memory=True)

    # 模型
    if args.method == 'film':
        model = FiLM_CALE(backbone_name=args.backbone).to(device)
        is_film = True
    else:
        model = CondBN_CALE(backbone_name=args.backbone).to(device)
        is_film = False

    # 类别权重
    all_labels = [lab for _, lab in train_items]
    counts = np.bincount(all_labels, minlength=7).astype(np.float32)
    counts[counts == 0] = 1.0
    cw = torch.tensor(1.0 / counts, dtype=torch.float32).to(device)
    cw = cw * (7.0 / cw.sum())
    print('class weights:', [f'{w:.3f}' for w in cw.cpu().numpy()])

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    best_val, best_state, best_ep = 0.0, None, -1
    wait = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss, n = 0.0, 0
        for imgs, labs, conds in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labs = labs.to(device, non_blocking=True)
            conds = conds.to(device, non_blocking=True)
            logits = model(imgs, conds)
            loss = F.cross_entropy(logits, labs, weight=cw)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            total_loss += loss.item(); n += 1
        sched.step()

        val_acc, val_uar, val_f1 = evaluate(model, val_loader, device, is_film)
        print(f'Epoch {epoch+1}/{args.epochs}: loss={total_loss/max(n,1):.4f} '
              f'val_acc={val_acc:.4f} val_uar={val_uar:.4f} val_f1={val_f1:.4f}',
              flush=True)

        if val_acc > best_val:
            best_val = val_acc
            best_ep = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f'Early stop at epoch {epoch+1}', flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, test_uar, test_f1 = evaluate(model, test_loader, device, is_film)
    print(f'\n=== Final ===')
    print(f'best_val_acc = {best_val:.4f} (epoch {best_ep})')
    print(f'test_acc     = {test_acc:.4f}')
    print(f'test_uar     = {test_uar:.4f}')
    print(f'test_f1      = {test_f1:.4f}')

    os.makedirs(args.out, exist_ok=True)
    ckpt = f'{args.out}/best_{args.method}_{args.backbone}_s{args.seed}.pth'
    torch.save(model.state_dict(), ckpt)
    print(f'Saved {ckpt}', flush=True)


if __name__ == '__main__':
    main()
