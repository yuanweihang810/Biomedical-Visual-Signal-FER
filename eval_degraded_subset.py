# paper2/eval_degraded_subset.py
"""RAF-DB 测试集上的退化子集评估：
   低亮度子集（brightness 下三分位）、模糊子集（Laplacian 方差下三分位）。
   报告 Baseline 和 Ours 在该子集上的 Top-1 准确率。"""
import os, argparse
import numpy as np
import torch
import cv2
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

def compute_stats(items, img_dir):
    """每个样本的亮度均值和 Laplacian 方差"""
    stats = []
    for fn, _ in items:
        p = os.path.join(img_dir, fn)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            stats.append((0.0, 0.0)); continue
        stats.append((float(img.mean()), float(cv2.Laplacian(img, cv2.CV_64F).var())))
    return np.array(stats)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--backbone', default='resnet18')
    ap.add_argument('--K', type=int, default=1)
    ap.add_argument('--no_transform', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, test_items, img_dir = load_rafdb('/root/autodl-tmp/IEEE')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    ds = DS(test_items, img_dir, tf)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=8)

    model = CALEPlus(backbone_name=args.backbone, K=args.K,
                     use_transform=not args.no_transform).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    all_preds = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            p, _ = model(imgs)
            all_preds.append(p.argmax(1).cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.array([lab for _, lab in test_items])

    stats = compute_stats(test_items, img_dir)
    bright, lap = stats[:,0], stats[:,1]

    # 下三分位
    b_thresh = np.quantile(bright, 1/3)
    l_thresh = np.quantile(lap, 1/3)
    low_bright = bright <= b_thresh
    low_res = lap <= l_thresh

    print(f'brightness threshold (1/3 quantile): {b_thresh:.2f}')
    print(f'Laplacian threshold (1/3 quantile): {l_thresh:.2f}')
    print(f'low-brightness subset size: {low_bright.sum()}')
    print(f'low-resolution subset size: {low_res.sum()}')
    print()
    print(f'Overall test acc:          {accuracy_score(labels, preds):.4f}')
    print(f'Low-brightness subset acc: {accuracy_score(labels[low_bright], preds[low_bright]):.4f}')
    print(f'Low-resolution subset acc: {accuracy_score(labels[low_res], preds[low_res]):.4f}')
