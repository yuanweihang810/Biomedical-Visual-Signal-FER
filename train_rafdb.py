# paper2/train_rafdb.py
import os, argparse, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
from cale_plus import CALEPlus
from rafdb_dataset import load_rafdb, RAFDBDataset


def build_loaders(args, train_items, val_items, test_items, img_dir):
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
    train_loader = DataLoader(RAFDBDataset(train_items, img_dir, train_tf),
                              batch_size=args.bs, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(RAFDBDataset(val_items, img_dir, eval_tf),
                            batch_size=args.bs, shuffle=False,
                            num_workers=8, pin_memory=True)
    test_loader = DataLoader(RAFDBDataset(test_items, img_dir, eval_tf),
                             batch_size=args.bs, shuffle=False,
                             num_workers=8, pin_memory=True)
    return train_loader, val_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(device)
            p, _ = model(imgs)
            preds.append(p.argmax(dim=1).cpu().numpy())
            gts.append(labs.numpy())
    preds = np.concatenate(preds, 0)
    gts = np.concatenate(gts, 0)
    return (accuracy_score(gts, preds),
            recall_score(gts, preds, average='macro', zero_division=0),
            f1_score(gts, preds, average='macro', zero_division=0))


def train(args, train_loader, val_loader, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CALEPlus(
        backbone_name=args.backbone,
        K=args.K,
        use_transform=not args.no_transform,
        transform_mode=args.transform_mode,
    ).to(device)

    # 类别权重
    all_labels = []
    for _, labs in train_loader:
        all_labels.extend(labs.numpy().tolist())
    counts = np.bincount(all_labels, minlength=7).astype(np.float32)
    counts[counts == 0] = 1.0
    class_weights = torch.tensor(1.0 / counts, dtype=torch.float32).to(device)
    class_weights = class_weights * (7.0 / class_weights.sum())
    print('class weights:', [f'{w:.3f}' for w in class_weights.cpu().numpy()], flush=True)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    best_val_acc, best_state, best_epoch = 0.0, None, -1
    patience, wait = args.patience, 0

    for epoch in range(args.epochs):
        model.train()
        total_loss, n = 0.0, 0
        for imgs, labs in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labs = labs.to(device, non_blocking=True)
            p, logits = model(imgs)
            K = logits.size(1)

            # 分支独立 CE
            loss_branch = sum(
                F.cross_entropy(logits[:, k, :], labs, weight=class_weights)
                for k in range(K)) / K
            # 聚合 CE
            loss_agg = F.cross_entropy(p, labs, weight=class_weights)
            loss = model.lambda_branch * loss_branch + model.lambda_agg * loss_agg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        scheduler.step()

        val_acc, val_uar, val_f1 = evaluate(model, val_loader, device)
        print(f'Epoch {epoch+1}/{args.epochs}: loss={total_loss/max(n,1):.4f} '
              f'val_acc={val_acc:.4f} val_uar={val_uar:.4f} val_f1={val_f1:.4f}',
              flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f'Early stop at epoch {epoch+1}', flush=True)
                break

    model.load_state_dict(best_state)
    test_acc, test_uar, test_f1 = evaluate(model, test_loader, device)
    return best_val_acc, test_acc, test_uar, test_f1, best_epoch, model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/root/autodl-tmp/IEEE')
    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--no_transform', action='store_true')
    parser.add_argument('--transform_mode', default='cond',
                        choices=['cond', 'rand'])
    parser.add_argument('--out', default='paper2/rafdb_exp')
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_all, test_items, img_dir = load_rafdb(args.data_root)
    print(f'RAF-DB: {len(train_all)} train | {len(test_items)} test', flush=True)

    labels = [lab for _, lab in train_all]
    train_items, val_items = train_test_split(
        train_all, test_size=args.val_ratio,
        random_state=args.seed, stratify=labels)
    print(f'Split: {len(train_items)} train | {len(val_items)} val', flush=True)

    train_loader, val_loader, test_loader = build_loaders(
        args, train_items, val_items, test_items, img_dir)

    best_val, test_acc, test_uar, test_f1, best_ep, model = train(
        args, train_loader, val_loader, test_loader)

    os.makedirs(args.out, exist_ok=True)
    if args.no_transform:
        tag = f'{args.backbone}_K{args.K}_notf_s{args.seed}'
    elif args.transform_mode == 'rand':
        tag = f'{args.backbone}_K{args.K}_rand_s{args.seed}'
    else:
        tag = f'{args.backbone}_K{args.K}_tf_s{args.seed}'
    ckpt = f'{args.out}/best_{tag}.pth'
    torch.save(model.state_dict(), ckpt)

    print(f'\n=== Final ===')
    print(f'best_val_acc = {best_val:.4f} (epoch {best_ep})')
    print(f'test_acc     = {test_acc:.4f}')
    print(f'test_uar     = {test_uar:.4f}')
    print(f'test_f1      = {test_f1:.4f}')
    print(f'Saved {ckpt}', flush=True)
