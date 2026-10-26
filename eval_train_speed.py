# paper2/eval_train_speed.py
import argparse, time
import torch
import torch.nn.functional as F
from cale_plus import CALEPlus

def measure_train(backbone, K, no_transform, bs, device,
                  n_warmup=3, n_iter=20, n_train_samples=11043):
    model = CALEPlus(backbone_name=backbone, K=K,
                     use_transform=not no_transform).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-3)
    x = torch.randn(bs, 3, 224, 224, device=device)
    y = torch.randint(0, 7, (bs,), device=device)
    def one_step():
        p, logits = model(x)
        K_ = logits.size(1)
        loss = sum(F.cross_entropy(logits[:, k, :], y) for k in range(K_)) / K_
        loss = loss + F.cross_entropy(p, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    for _ in range(n_warmup): one_step()
    if device.type == 'cuda': torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter): one_step()
    if device.type == 'cuda': torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    per_batch_ms = dt / n_iter * 1000
    n_batches = n_train_samples // bs
    per_epoch_s = per_batch_ms / 1000 * n_batches
    n_params = sum(p.numel() for p in model.parameters())
    return n_params, per_batch_ms, per_epoch_s, n_batches

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n_iter', type=int, default=20)
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f'Train | {args.backbone} | bs={args.bs} | {args.device} | '
          f'samples/epoch=11043')
    print('-' * 88)
    print(f'{"Config":<22} {"Params(M)":>10} {"per-batch(ms)":>15} '
          f'{"per-epoch(s)":>14} {"batches":>10} {"Rel.":>8}')
    print('-' * 88)
    base = None
    for cfg, K, no_tf in [('Baseline (K=1)', 1, True), ('Ours (K=4)', 4, False)]:
        p, bms, eps, nb = measure_train(args.backbone, K, no_tf, args.bs, device,
                                        n_iter=args.n_iter)
        rel = '' if base is None else f'{eps/base:.2f}x'
        if base is None: base = eps
        print(f'{cfg:<22} {p/1e6:>10.2f} {bms:>15.2f} {eps:>14.1f} '
              f'{nb:>10} {rel:>8}', flush=True)
