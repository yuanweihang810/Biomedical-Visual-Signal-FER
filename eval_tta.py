# paper2/eval_tta.py
"""ResNet-50 单分支 + 水平翻转 TTA，作为同等 FLOPs 对照。"""
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
    def __init__(self, items, img_dir):
        self.items = items; self.img_dir = img_dir
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        fn, lab = self.items[idx]
        img = Image.open(os.path.join(self.img_dir, fn)).convert('RGB')
        return transforms.ToTensor()(transforms.Resize((224,224))(img)), lab


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--backbone', default='resnet50')
    args = ap.parse_args()
    device = torch.device('cuda')

    _, test_items, img_dir = load_rafdb('/root/autodl-tmp/IEEE')
    loader = DataLoader(DS(test_items, img_dir), batch_size=48,
                        shuffle=False, num_workers=8)

    model = CALEPlus(backbone_name=args.backbone, K=1,
                     use_transform=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    preds, gts = [], []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(device)
            # 原图 + 翻转
            flip = torch.flip(imgs, dims=[3])
            # 模型内部已做 normalize，直接前向
            p1, _ = model(imgs)
            p2, _ = model(flip)
            p = (torch.softmax(p1, -1) + torch.softmax(p2, -1)) / 2
            preds.append(p.argmax(1).cpu().numpy())
            gts.append(labs.numpy())
    preds = np.concatenate(preds); gts = np.concatenate(gts)
    print(f'ResNet-50 + HFlip TTA: test_acc = {accuracy_score(gts, preds):.4f}')
