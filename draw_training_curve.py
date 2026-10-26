import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

def parse_log(path):
    eps, accs = [], []
    with open(path) as f:
        for line in f:
            m = re.search(r'Epoch (\d+)/\d+:.*val_acc=([\d.]+)', line)
            if m:
                eps.append(int(m.group(1)))
                accs.append(float(m.group(2)))
    return eps, accs

ep_ours, acc_ours = parse_log('/root/autodl-tmp/IEEE/paper2/logs/resnet18_ours_new.log')
ep_base, acc_base = parse_log('/root/autodl-tmp/IEEE/paper2/logs/resnet18_baseline_80.log')

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(ep_base, acc_base, label='Baseline (K=1)', color='#26A69A', lw=1.5)
ax.plot(ep_ours, acc_ours, label='Ours (K=4)', color='#EC407A', lw=1.5)
ax.axvline(x=60, color='gray', linestyle='--', alpha=0.6, label='epoch 60')

a60 = next((acc_ours[i] for i, e in enumerate(ep_ours) if e == 60), None)
if a60:
    ax.annotate(f'{a60:.4f}', xy=(60, a60), xytext=(62, a60-0.005),
                fontsize=10, color='#EC407A',
                arrowprops=dict(arrowstyle='->', color='#EC407A'))

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Validation accuracy', fontsize=12)
ax.set_xlim(0, 80)
ax.grid(alpha=0.3, linestyle='--')
ax.legend(fontsize=11, loc='lower right')
plt.tight_layout()
plt.savefig('/root/autodl-tmp/IEEE/paper2/fig_training_curve.pdf', bbox_inches='tight')
print('fig_training_curve.pdf saved')

if a60:
    best = max(acc_ours)
    best_ep = ep_ours[acc_ours.index(best)]
    print(f'epoch 60 val_acc: {a60:.4f}')
    print(f'best val_acc: {best:.4f} at epoch {best_ep}')
    print(f'gap: {best - a60:.4f}')
