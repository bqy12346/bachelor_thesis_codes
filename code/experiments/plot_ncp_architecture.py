"""
plot_cfc_ncp_wiring.py — 精确对应 cfc_ncp_model.py 的结构可视化
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from ncps.wirings import NCP

# ── 完全照抄 cfc_ncp_model.py 里的参数 ──────────────────────────────────────
MOTOR_NEURONS = 64          # run_superdiagnostic.py 中传入的值
CNN_OUT_CH    = 64          # CfC 的 input_size，即 sensory 输入维度
N_CLASSES     = 5           # superdiagnostic 任务的输出类别数

wiring = NCP(
    inter_neurons=16,
    command_neurons=8,
    motor_neurons=MOTOR_NEURONS,
    sensory_fanout=8,
    inter_fanout=4,
    recurrent_command_synapses=4,
    motor_fanin=4,
)
wiring.build(CNN_OUT_CH)   # sensory = 64

# ── 打印结构摘要 ──────────────────────────────────────────────────────────────
total_neurons  = wiring.units
hidden_neurons = total_neurons - wiring.output_dim
synapses       = int((abs(wiring.adjacency_matrix) > 0).sum())

print("=" * 50)
print("  CfC-NCP Wiring Summary")
print("=" * 50)
print(f"  [CNN frontend]  12 → 32 → 64 ch,  T: 1000→250→50")
print(f"  Sensory neurons : {wiring.input_dim:>4}  (= CNN_OUT_CH)")
print(f"  Inter   neurons : {16:>4}")
print(f"  Command neurons : {8:>4}")
print(f"  Motor   neurons : {MOTOR_NEURONS:>4}  (= motor_neurons param)")
print(f"  Total   neurons : {total_neurons:>4}  (inter+command+motor)")
print(f"  Total   synapses: {synapses:>4}")
print(f"  [FC head]       {MOTOR_NEURONS} → {N_CLASSES}  (n_classes)")
print("=" * 50)

# ── 绘图 ──────────────────────────────────────────────────────────────────────
COLORS = {
    "sensory": "#7B2D8B",
    "inter":   "#E65100",
    "command": "#00838F",
    "motor":   "#2E7D32",
}

fig = plt.figure(figsize=(20, 9))
fig.patch.set_facecolor("#F8F8F8")

# 左：NCP wiring graph（spiral）
ax1 = fig.add_subplot(1, 2, 1)
sns.set_style("white")
plt.sca(ax1)
handles = wiring.draw_graph(layout="spiral", neuron_colors=COLORS)
ax1.legend(handles=handles, loc="upper right", fontsize=9,
           title="Neuron type", framealpha=0.8)
ax1.set_title(
    f"CfC-NCP Wiring  (spiral layout)\n"
    f"Sensory={wiring.input_dim}  Inter=16  Command=8  Motor={MOTOR_NEURONS}",
    fontsize=11, pad=10,
)

# 右：整体模型结构示意图（手绘 block diagram）
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis("off")
ax2.set_facecolor("#F8F8F8")
ax2.set_title("CfC-NCP Full Model Architecture", fontsize=11, pad=10)

BLOCK_COLORS = ["#CFD8DC", "#B3E5FC", "#C8E6C9", "#FFF9C4", "#FFCCBC", "#E1BEE7"]
blocks = [
    (1.0, 7.8, 8.0, 1.0, BLOCK_COLORS[0],
     "Input ECG\n(B, T=1000, 12)"),
    (1.5, 5.9, 7.0, 1.4, BLOCK_COLORS[1],
     "CNN Frontend\nConv1d(12→32, k=5, s=4)  +  BN  +  GELU\n"
     "Conv1d(32→64, k=5, s=5)  +  BN  +  GELU\n→ (B, T=50, 64)"),
    (1.5, 3.5, 7.0, 1.9, BLOCK_COLORS[2],
     f"CfC-NCP  (mixed_memory={True})\n"
     f"Sensory=64  Inter=16  Command=8  Motor={MOTOR_NEURONS}\n"
     f"Total hidden = {total_neurons} neurons,  {synapses} synapses\n"
     f"→ take last step → (B, {MOTOR_NEURONS})"),
    (1.5, 2.1, 7.0, 0.9, BLOCK_COLORS[3],
     f"FC Head:  Linear({MOTOR_NEURONS} → {N_CLASSES})"),
    (1.0, 0.8, 8.0, 0.9, BLOCK_COLORS[4],
     f"Output logits  (B, {N_CLASSES})   +   BCEWithLogitsLoss"),
]

for (x, y, w, h, color, label) in blocks:
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor="#607D8B", linewidth=1.5,
    )
    ax2.add_patch(rect)
    ax2.text(x + w / 2, y + h / 2, label,
             ha="center", va="center", fontsize=8.5,
             fontfamily="monospace", wrap=True)

# 箭头
arrow_kw = dict(arrowstyle="-|>", color="#37474F", lw=1.5)
for y_tail, y_head in [(7.8, 7.3), (5.9, 5.4), (3.5, 3.0), (2.1, 1.7)]:
    ax2.annotate("", xy=(5, y_head), xytext=(5, y_tail),
                 arrowprops=arrow_kw)

sns.despine(left=True, bottom=True)
plt.tight_layout()

OUT = "/scratch2/bsc26f19/projects/bachelor_thesis_codes/cfc_ncp_architecture.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\n  [Saved] {OUT}")