# paper2/eval_mechanism.py
import os, json, argparse, glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.metrics import f1_score, average_precision_score
from cale_plus import CALEPlus


class MLTestDataset(Dataset):
    def __init__(self, img_dir, label_file, cond_file, tf):
        self.img_dir = img_dir
        self.tf = tf
        self.items = []
        labels_map = {}
        with open(label_file) as f:
            lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            labels_map[parts[0]] = list(map(int, parts[1:8]))
        cond_df = pd.read_csv(cond_file)
        cond_map = {row['filename'].rsplit('.', 1)[0]: int(row['condition'])
                    for _, row in cond_df.iterrows()}
        for fn in sorted(os.listdir(img_dir)):
            stem = fn.rsplit('.', 1)[0]
            if stem in labels_map and stem in cond_map:
                self.items.append((fn, labels_map[stem], cond_map[stem]))
        print(f'MLTestDataset: matched {len(self.items)}')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fn, lab, cond = self.items[idx]
        img = Image.open(os.path.join(self.img_dir, fn)).convert('RGB')
        if self.tf:
            img = self.tf(img)
        return img, torch.tensor(lab, dtype=torch.float32), cond


def ece_score(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i + 1])
        if mask.sum() > 0:
            e += mask.sum() / len(p) * abs(y[mask].mean() - p[mask].mean())
    return float(e)


def evaluate(model, loader, device, target_k=2.5):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for imgs, lab, cond in loader:
            imgs = imgs.to(device)
            cond = cond.to(device)
            p_agg, _, _, _ = model(imgs, cond)
            probs.append(torch.sigmoid(p_agg).cpu().numpy())
            labels.append(lab.numpy())
    probs = np.concatenate(probs, 0)
    labels = np.concatenate(labels, 0)

    ap = [average_precision_score(labels[:, i], probs[:, i])
          for i in range(labels.shape[1])]
    mAP = float(np.mean(ap))

    k = int(round(target_k))
    preds = np.zeros_like(probs, dtype=int)
    topk_idx = np.argsort(-probs, axis=1)[:, :k]
    for i in range(preds.shape[0]):
        preds[i, topk_idx[i]] = 1
    macro_f1_topk = float(f1_score(labels, preds, average='macro', zero_division=0))
    avg_labels_topk = float(preds.sum(axis=1).mean())

    preds_05 = (probs > 0.5).astype(int)
    macro_f1_05 = float(f1_score(labels, preds_05, average='macro', zero_division=0))
    avg_labels_05 = float(preds_05.sum(axis=1).mean())

    ece = ece_score(labels.flatten(), probs.flatten())
    return dict(macro_f1_topk=macro_f1_topk, avg_labels_topk=avg_labels_topk,
                macro_f1_05=macro_f1_05, avg_labels_05=avg_labels_05,
                mAP=mAP, ece=ece)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', default='paper2/exp')
    parser.add_argument('--agg_list', default='A0,A1,A2,A3,A4')
    parser.add_argument('--ratio_list', default='1.0')
    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument('--out', default='paper2/mechanism_results.json')
    parser.add_argument('--target_k', type=float, default=2.5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    test_ds = MLTestDataset('/root/autodl-tmp/IEEE/test_aligned/test',
                            '/root/autodl-tmp/IEEE/label.txt',
                            '/root/autodl-tmp/IEEE/paper2/test_condition_labels.csv',
                            tf)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    results = {}
    for agg in args.agg_list.split(','):
        results[agg] = {}
        for ratio in args.ratio_list.split(','):
            pattern = f'{args.exp_dir}/best_{args.backbone}_{agg}_r{ratio}_f*_s*.pth'
            ckpts = sorted(glob.glob(pattern))
            if not ckpts:
                results[agg][ratio] = None
                print(f'[skip] no ckpt: {pattern}')
                continue
            fold_metrics = []
            for ckpt in ckpts:
                model = CALEPlus(backbone_name=args.backbone, agg_type=agg).to(device)
                model.load_state_dict(torch.load(ckpt, map_location=device))
                m = evaluate(model, test_loader, device, args.target_k)
                fold_metrics.append(m)
                print(f'{os.path.basename(ckpt)}: {m}', flush=True)
            agg_summary = {}
            for kk in fold_metrics[0]:
                vals = [m[kk] for m in fold_metrics]
                agg_summary[kk + '_mean'] = float(np.mean(vals))
                agg_summary[kk + '_std'] = float(np.std(vals))
            results[agg][ratio] = agg_summary

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    print('\n' + '=' * 110)
    print(f'{"agg":<6} {"AvgLabels(topk)":>18} {"F1(topk)":>18} '
          f'{"AvgLabels(0.5)":>18} {"F1(0.5)":>18} {"mAP":>18} {"ECE":>18}')
    print('-' * 110)
    for agg in args.agg_list.split(','):
        for ratio in args.ratio_list.split(','):
            r = results[agg].get(ratio)
            if r is None:
                continue
            print(f'{agg:<6} '
                  f'{r["avg_labels_topk_mean"]:>8.3f}±{r["avg_labels_topk_std"]:<7.3f} '
                  f'{r["macro_f1_topk_mean"]:>8.4f}±{r["macro_f1_topk_std"]:<7.4f} '
                  f'{r["avg_labels_05_mean"]:>8.3f}±{r["avg_labels_05_std"]:<7.3f} '
                  f'{r["macro_f1_05_mean"]:>8.4f}±{r["macro_f1_05_std"]:<7.4f} '
                  f'{r["mAP_mean"]:>8.4f}±{r["mAP_std"]:<7.4f} '
                  f'{r["ece_mean"]:>8.4f}±{r["ece_std"]:<7.4f}')
    print('=' * 110)
