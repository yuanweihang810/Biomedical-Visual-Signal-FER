# paper2/eval_mean_agg.py
import os, argparse
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score
from cale_plus import CALEPlus
from rafdb_dataset import load_rafdb


class DS(Dataset):
    def __init__(self, items, img_dir, tf):
        self.items = items; self.img_dir = img_dir; self.tf = tf
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        fn, lab = self.items[idx]
        img = Image.open(os.path.join(self.img_dir, fn)).convert('RGB')
        if self.tf: img = self.tf(img)
        return img, lab


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--backbone', default='resnet18')
    args = ap.parse_args()
    device = torch.device('cuda')

    _, test_items, img_dir = load_rafdb('/root/autodl-tmp/IEEE')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    loader = DataLoader(DS(test_items, img_dir, tf), batch_size=64,
                        shuffle=False, num_workers=8)

    model = CALEPlus(backbone_name=args.backbone, K=4).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    for mode in ['mlp', 'mean']:
        preds, gts = [], []
        with torch.no_grad():
            for imgs, labs in loader:
                imgs = imgs.to(device)
                logits_list = []
                for k in range(model.K):
                    x_k = model.transform(imgs, k)
                    x_k = (x_k - model.imagenet_mean) / model.imagenet_std
                    feat = model.backbone(x_k)
                    feat = model.branch_bns[k](feat)
                    logits_list.append(model.branch_heads[k](feat))
                logits = torch.stack(logits_list, dim=1)
                if mode == 'mlp':
                    flat = logits.flatten(1)
                    w = torch.softmax(model.agg_mlp(flat), dim=1)
                    p = (w.unsqueeze(-1) * logits).sum(dim=1)
                else:
                    p = logits.mean(dim=1)
                preds.append(p.argmax(1).cpu().numpy())
                gts.append(labs.numpy())
        preds = np.concatenate(preds); gts = np.concatenate(gts)
        acc = accuracy_score(gts, preds)
        print(f'agg={mode}: test_acc = {acc:.4f}')
