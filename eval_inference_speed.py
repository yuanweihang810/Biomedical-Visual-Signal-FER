# paper2/eval_inference_speed.py
import argparse, time
import torch
from cale_plus import CALEPlus

try:
    from thop import profile as thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

def measure(backbone, K, no_transform, device, bs=1, n_warmup=20, n_iter=100):
    model = CALEPlus(backbone_name=backbone, K=K,
                     use_transform=not no_transform).to(device)
    model.eval()
    x = torch.randn(bs, 3, 224, 224, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    flops_g = float('nan')
    if HAS_THOP:
        flops, _ = thop_profile(model, inputs=(x,), verbose=False)
        flops_g = flops / 1e9
    with torch.no_grad():
        for _ in range(n_warmup): model(x)
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter): model(x)
        if device.type == 'cuda': torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) / n_iter * 1000
    return n_params, flops_g, dt_ms

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n_iter', type=int, default=100)
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f'Inference | {args.backbone} | bs={args.bs} | {args.device}')
    print('-' * 78)
    print(f'{"Config":<22} {"Params(M)":>10} {"FLOPs(G)":>10} '
          f'{"Latency(ms)":>14} {"Rel.":>8}')
    print('-' * 78)
    base = None
    for cfg, K, no_tf in [('Baseline (K=1)', 1, True), ('Ours (K=4)', 4, False)]:
        p, f, t = measure(args.backbone, K, no_tf, device, args.bs, n_iter=args.n_iter)
        rel = '' if base is None else f'{t/base:.2f}x'
        if base is None: base = t
        print(f'{cfg:<22} {p/1e6:>10.2f} {f:>10.2f} {t:>14.2f} {rel:>8}', flush=True)
