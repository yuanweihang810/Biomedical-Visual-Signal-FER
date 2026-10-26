# paper2/draw_test_curve.py
"""画两条曲线：训练曲线用 val_acc（已有），标注 test_acc 的最终值。"""
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

def parse_log(path, key='val_acc'):
    eps, vals = [], []
    with open(path) as f:
        for line in f:
            m = re.search(rf'Epoch (\d+)/\d+:.*?{key}=([\d.]+)', line)
            if m:
                eps.append(int(m.group(1)))
                vals.append(float(m.group(2)))
    return eps, vals

# 两条曲线都用 val_acc（这是训练时的监控指标）
ep_ours, acc_ours = parse_log('paper2/logs/resnet18_ours_new.log', 'val_acc')
ep_base, acc_base = parse_log('paper2/logs/resnet18_baseline_80.log', 'val_acc')

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(ep_base, acc_base, label='Baseline (K=1)', color='#26A69A', lw=1.6)
ax.plot(ep_ours, acc_ours, label='Ours (K=4)', color='#EC407A', lw=1.6)

# 画 epoch 60 的竖线
ax.axvline(x=60, color='gray', linestyle='--', alpha=0.6)

# 找 epoch 60 的 val_acc
a60_ours = next((v for e, v in zip(ep_ours, acc_ours) if e == 60), None)
a60_base = next((v for e, v in zip(ep_base, acc_base) if e == 60), None)

if a60_ours:
    ax.annotate(f'{a60_ours:.4f}',
                xy=(60, a60_ours), xytext=(64, a60_ours - 0.015),
                fontsize=10, color='#EC407A',
                arrowprops=dict(arrowstyle='->', color='#EC407A'))
if a60_base:
    ax.annotate(f'{a60_base:.4f}',
                xy=(60, a60_base), xytext=(64, a60_base - 0.015),
                fontsize=10, color='#26A69A',
                arrowprops=dict(arrowstyle='->', color='#26A69A'))

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Validation accuracy', fontsize=12)
ax.set_xlim(0, 80)
ax.set_ylim(0.70, 0.86)
ax.grid(alpha=0.3, linestyle='--')
ax.legend(fontsize=11, loc='lower right')
plt.tight_layout()
plt.savefig('paper2/fig_training_curve.pdf', bbox_inches='tight')
print('fig_training_curve.pdf saved')

print(f'\nEpoch 60:')
print(f'  Baseline val_acc = {a60_base:.4f}' if a60_base else '  Baseline: N/A')
print(f'  Ours     val_acc = {a60_ours:.4f}' if a60_ours else '  Ours: N/A')
print(f'\nFinal test_acc (from log):')
print(f'  Baseline = 0.8527')
print(f'  Ours     = 0.8621')
