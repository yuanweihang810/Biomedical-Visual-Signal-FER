# paper2/eval_mechanism.py
"""
E2 机制性预测检验：
预测1：归一化聚合(A4)应在所有数据规模下把 AvgLabels 稳定在目标值 ±0.3 内。
预测2：可学习标量(A1)部分抑制漂移。
预测3：固定求和(A0)的 AvgLabels 方差应随数据规模显著变化。
"""
import os, json, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.metrics import f1_score, average_precision_score
from cale_plus import CALEPlus


class MLTestDataset(Dataset):
    def __init__(self, img_dir, label_file, tf):
        self.img_dir = img_dir; self.tf = tf
        self.items = []
        with open(label_file) as f:
            for line in f:
                p = line.strip().split()
                if len(p) < 8: continue
                self.items.append((p[0], list(map(int, p[1:8]))))
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        fn, lab = self.items[idx]
        img = Image.open(os.path.join(self.img_dir, fn)).convert('RGB')
        if self.tf: img = self.tf(img)
        return img, torch.tensor(lab, dtype=torch.float32)


def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for imgs, lab in loader:
            imgs = imgs.to(device)
            # 测试时条件未知，用随机条件作为下界；主设定中用估计器
            B = imgs.shape[0]
            conds = torch.randint(0, 4, (B,), device=device)
            p, _, _, _ = model(imgs, conds)
            probs.append(p.cpu().numpy()); labels.append(lab.numpy())
    probs = np.concatenate(probs, 0); labels = np.concatenate(labels, 0)
    preds = (probs > 0.5).astype(int)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    avg_labels = preds.sum(axis=1).mean()
    mAP = np.mean([average_precision_score(labels[:, i], probs[:, i]) for i in range(labels.shape[1])])
    return dict(macro_f1=macro_f1, avg_labels=float(avg_labels), mAP=float(mAP))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', default='paper2/exp')
    parser.add_argument('--agg_list', default='A0,A1,A2,A3,A4')
    parser.add_argument('--ratio_list', default='0.1,0.25,0.5,1.0')
    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument('--out', default='paper2/mechanism_results.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    test_ds = MLTestDataset('/root/autodl-tmp/IEEE/test_aligned/test',
                            '/root/autodl-tmp/IEEE/label.txt', tf)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    results = {}
    for agg in args.agg_list.split(','):
        results[agg] = {}
        for ratio in args.ratio_list.split(','):
            # 这里只取 fold0 seed42，完整实验需遍历所有折与种子
            ckpt = f'{args.exp_dir}/best_{args.backbone}_{agg}_r{ratio}_f0_s42.pth'
            if not os.path.exists(ckpt):
                results[agg][ratio] = None; continue
            model = CALEPlus(backbone_name=args.backbone, agg_type=agg).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            results[agg][ratio] = evaluate(model, test_loader, device)

    # 机制性预测检验
    target_avg = 2.5
    verdict = {}
    for agg in results:
        vals = [results[agg][r]['avg_labels'] for r in args.ratio_list.split(',') if results[agg][r]]
        if not vals: continue
        drift = max(vals) - min(vals)
        verdict[agg] = dict(drift=drift,
                            within_0_3=all(abs(v - target_avg) <= 0.3 for v in vals))
    results['verdict'] = verdict
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(verdict, indent=2))