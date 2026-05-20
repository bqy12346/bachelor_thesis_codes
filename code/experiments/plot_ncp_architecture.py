#  螺旋结构
# """
# plot_cfc_ncp_wiring.py — 精确对应 cfc_ncp_model.py 的结构可视化
# """
# import os
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import seaborn as sns
# from ncps.wirings import NCP

# # ── 完全照抄 cfc_ncp_model.py 里的参数 ──────────────────────────────────────
# MOTOR_NEURONS = 64          # run_superdiagnostic.py 中传入的值
# CNN_OUT_CH    = 64          # CfC 的 input_size，即 sensory 输入维度
# N_CLASSES     = 5           # superdiagnostic 任务的输出类别数

# wiring = NCP(
#     inter_neurons=16,
#     command_neurons=8,
#     motor_neurons=MOTOR_NEURONS,
#     sensory_fanout=8,
#     inter_fanout=4,
#     recurrent_command_synapses=4,
#     motor_fanin=4,
# )
# wiring.build(CNN_OUT_CH)   # sensory = 64

# # ── 打印结构摘要 ──────────────────────────────────────────────────────────────
# total_neurons  = wiring.units
# hidden_neurons = total_neurons - wiring.output_dim
# synapses       = int((abs(wiring.adjacency_matrix) > 0).sum())

# print("=" * 50)
# print("  CfC-NCP Wiring Summary")
# print("=" * 50)
# print(f"  [CNN frontend]  12 → 32 → 64 ch,  T: 1000→250→50")
# print(f"  Sensory neurons : {wiring.input_dim:>4}  (= CNN_OUT_CH)")
# print(f"  Inter   neurons : {16:>4}")
# print(f"  Command neurons : {8:>4}")
# print(f"  Motor   neurons : {MOTOR_NEURONS:>4}  (= motor_neurons param)")
# print(f"  Total   neurons : {total_neurons:>4}  (inter+command+motor)")
# print(f"  Total   synapses: {synapses:>4}")
# print(f"  [FC head]       {MOTOR_NEURONS} → {N_CLASSES}  (n_classes)")
# print("=" * 50)

# # ── 绘图 ──────────────────────────────────────────────────────────────────────
# COLORS = {
#     "sensory": "#7B2D8B",
#     "inter":   "#E65100",
#     "command": "#00838F",
#     "motor":   "#2E7D32",
# }

# fig = plt.figure(figsize=(20, 9))
# fig.patch.set_facecolor("#F8F8F8")

# # 左：NCP wiring graph（spiral）
# ax1 = fig.add_subplot(1, 2, 1)
# sns.set_style("white")
# plt.sca(ax1)
# handles = wiring.draw_graph(layout="spiral", neuron_colors=COLORS)
# ax1.legend(handles=handles, loc="upper right", fontsize=9,
#            title="Neuron type", framealpha=0.8)
# ax1.set_title(
#     f"CfC-NCP Wiring  (spiral layout)\n"
#     f"Sensory={wiring.input_dim}  Inter=16  Command=8  Motor={MOTOR_NEURONS}",
#     fontsize=11, pad=10,
# )

# # 右：整体模型结构示意图（手绘 block diagram）
# ax2 = fig.add_subplot(1, 2, 2)
# ax2.set_xlim(0, 10)
# ax2.set_ylim(0, 10)
# ax2.axis("off")
# ax2.set_facecolor("#F8F8F8")
# ax2.set_title("CfC-NCP Full Model Architecture", fontsize=11, pad=10)

# BLOCK_COLORS = ["#CFD8DC", "#B3E5FC", "#C8E6C9", "#FFF9C4", "#FFCCBC", "#E1BEE7"]
# blocks = [
#     (1.0, 7.8, 8.0, 1.0, BLOCK_COLORS[0],
#      "Input ECG\n(B, T=1000, 12)"),
#     (1.5, 5.9, 7.0, 1.4, BLOCK_COLORS[1],
#      "CNN Frontend\nConv1d(12→32, k=5, s=4)  +  BN  +  GELU\n"
#      "Conv1d(32→64, k=5, s=5)  +  BN  +  GELU\n→ (B, T=50, 64)"),
#     (1.5, 3.5, 7.0, 1.9, BLOCK_COLORS[2],
#      f"CfC-NCP  (mixed_memory={True})\n"
#      f"Sensory=64  Inter=16  Command=8  Motor={MOTOR_NEURONS}\n"
#      f"Total hidden = {total_neurons} neurons,  {synapses} synapses\n"
#      f"→ take last step → (B, {MOTOR_NEURONS})"),
#     (1.5, 2.1, 7.0, 0.9, BLOCK_COLORS[3],
#      f"FC Head:  Linear({MOTOR_NEURONS} → {N_CLASSES})"),
#     (1.0, 0.8, 8.0, 0.9, BLOCK_COLORS[4],
#      f"Output logits  (B, {N_CLASSES})   +   BCEWithLogitsLoss"),
# ]

# for (x, y, w, h, color, label) in blocks:
#     rect = mpatches.FancyBboxPatch(
#         (x, y), w, h,
#         boxstyle="round,pad=0.1",
#         facecolor=color, edgecolor="#607D8B", linewidth=1.5,
#     )
#     ax2.add_patch(rect)
#     ax2.text(x + w / 2, y + h / 2, label,
#              ha="center", va="center", fontsize=8.5,
#              fontfamily="monospace", wrap=True)

# # 箭头
# arrow_kw = dict(arrowstyle="-|>", color="#37474F", lw=1.5)
# for y_tail, y_head in [(7.8, 7.3), (5.9, 5.4), (3.5, 3.0), (2.1, 1.7)]:
#     ax2.annotate("", xy=(5, y_head), xytext=(5, y_tail),
#                  arrowprops=arrow_kw)

# sns.despine(left=True, bottom=True)
# plt.tight_layout()

# OUT = "/scratch2/bsc26f19/projects/bachelor_thesis_codes/cfc_ncp_architecture.png"
# os.makedirs(os.path.dirname(OUT), exist_ok=True)
# plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
# plt.close()
# print(f"\n  [Saved] {OUT}")



"""
plot_cfc_ncp_wiring.py  —  真实训练权重版（方案 B）
直接从 CfC-NCP 的三层 wired cell 提取每条连接的真实权重 (ff1.weight * sparsity_mask)
线条粗细 + 颜色深浅 = 该连接训练后权重的绝对值

CfC-NCP 三层结构（已从权重 shape 解码确认）：
  layer_0: [64 sensory + 32 inter(self)] -> 32 inter      ff1 (32, 96)
  layer_1: [32 inter   + 16 cmd(self)]   -> 16 command    ff1 (16, 48)
  layer_2: [16 command + 64 motor(self)] -> 64 motor      ff1 (64, 80)
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors

# ══════════════════════════════════════════════════════════════════════════════
#  ★ 修改这里 ★
PT_FILE = "/scratch2/bsc26f19/projects/bachelor_thesis_codes/code/saved_models/cfc_ncp_stft_20260519_114735.pt"
TOP_K_PER_PAIR = 60      # 每对层最多显示的连线数（按权重取最大）
# ══════════════════════════════════════════════════════════════════════════════

OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(PT_FILE)),
                        "cfc_ncp_wiring_weights.png")

SENSORY_N = 64
INTER_N   = 32
COMMAND_N = 16
MOTOR_N   = 64
N_CLASSES = 5

# ── 1. 加载权重，提取每层有效连接矩阵 ────────────────────────────────────────
state = torch.load(PT_FILE, map_location="cpu")

def eff_weight(li):
    """该层有效权重 = ff1.weight * sparsity_mask，返回绝对值矩阵 (out, in)"""
    ff1  = state[f"rnn.rnn_cell.layer_{li}.ff1.weight"].numpy()
    mask = state[f"rnn.rnn_cell.layer_{li}.sparsity_mask"].numpy()
    return np.abs(ff1 * mask)

W0 = eff_weight(0)   # (32, 96)  -> inter，   输入 [0:64]=sensory, [64:96]=inter(self)
W1 = eff_weight(1)   # (16, 48)  -> command, 输入 [0:32]=inter,   [32:48]=command(self)
W2 = eff_weight(2)   # (64, 80)  -> motor,   输入 [0:16]=command, [16:80]=motor(self)

# 全局归一化，所有连线用同一把尺子
gmax = max(W0.max(), W1.max(), W2.max()) + 1e-9

# 各层连接：(src_local, dst_local, weight_normalized)
# S→I : W0[:, 0:64]   (dst=inter 32, src=sensory 64)
SI = [(s, d, W0[d, s] / gmax)
      for d in range(INTER_N) for s in range(SENSORY_N)
      if W0[d, s] > 0]
# I→C : W1[:, 0:32]   (dst=command 16, src=inter 32)
IC = [(s, d, W1[d, s] / gmax)
      for d in range(COMMAND_N) for s in range(INTER_N)
      if W1[d, s] > 0]
# C→M : W2[:, 0:16]   (dst=motor 64, src=command 16)
CM = [(s, d, W2[d, s] / gmax)
      for d in range(MOTOR_N) for s in range(COMMAND_N)
      if W2[d, s] > 0]
# C→C : W1[:, 32:48]  command 自递归 (dst=command 16, src=command 16)
CC = [(s, d, W1[d, 32 + s] / gmax)
      for d in range(COMMAND_N) for s in range(COMMAND_N)
      if W1[d, 32 + s] > 0]

print(f"S->I edges: {len(SI)}")
print(f"I->C edges: {len(IC)}")
print(f"C->C edges: {len(CC)}")
print(f"C->M edges: {len(CM)}")

# 节点重要性（用于节点颜色深浅）= 每个节点所有出入连接权重之和
def node_imp(n, edges_in, edges_out):
    imp = np.zeros(n)
    for (s, d, w) in edges_in:
        imp[d] += w
    for (s, d, w) in edges_out:
        imp[s] += w
    return imp / (imp.max() + 1e-9)

imp_sensory = node_imp(SENSORY_N, [], SI)
imp_inter   = node_imp(INTER_N,   SI, IC)
imp_command = node_imp(COMMAND_N, IC + CC, CM + CC)
# motor 重要性用 fc.weight（直接衡量对分类输出的贡献）
fc_w = state["fc.weight"].numpy()                 # (5, 64)
imp_motor = np.linalg.norm(fc_w, axis=0)
imp_motor /= imp_motor.max() + 1e-9

# ── 2. 坐标 ───────────────────────────────────────────────────────────────────
LAYER_X = [0.10, 0.35, 0.62, 0.87]

def make_pos(n, x):
    return np.array([(x, y) for y in np.linspace(0.94, 0.06, n)])

pos_s = make_pos(SENSORY_N, LAYER_X[0])
pos_i = make_pos(INTER_N,   LAYER_X[1])
pos_c = make_pos(COMMAND_N, LAYER_X[2])
pos_m = make_pos(MOTOR_N,   LAYER_X[3])

# ── 3. 绘图 ───────────────────────────────────────────────────────────────────
COLORS_NODE = {
    "sensory": "#7B2D8B",
    "inter":   "#E65100",
    "command": "#00838F",
    "motor":   "#2E7D32",
}
PAIR_COLORS = {
    "S→I": "#E65100",
    "I→C": "#00838F",
    "C→C": "#006064",
    "C→M": "#2E7D32",
}

fig = plt.figure(figsize=(20, 9))
fig.patch.set_facecolor("#F8F8F8")

ax1 = fig.add_subplot(1, 2, 1)
ax1.set_facecolor("#F8F8F8"); ax1.axis("off")
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
ax1.set_title(
    f"CfC-NCP Wiring  (trained weights)\n"
    f"Sensory={SENSORY_N}  Inter={INTER_N}  Command={COMMAND_N}  Motor={MOTOR_N}",
    fontsize=11, pad=10,
)

def draw_group(ax, edges, src_pos, dst_pos, color, top_k, dashed=False):
    if not edges:
        return
    top = sorted(edges, key=lambda e: e[2], reverse=True)[:top_k]
    ls  = "--" if dashed else "-"
    for (si, di, w) in top:
        p0, p1 = src_pos[si], dst_pos[di]
        w = float(np.clip(w, 0, 1))
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                color=color, alpha=0.15 + 0.65 * w,
                linewidth=0.3 + 2.2 * w, linestyle=ls, zorder=1)

draw_group(ax1, SI, pos_s, pos_i, PAIR_COLORS["S→I"], TOP_K_PER_PAIR)
draw_group(ax1, IC, pos_i, pos_c, PAIR_COLORS["I→C"], TOP_K_PER_PAIR)
draw_group(ax1, CC, pos_c, pos_c, PAIR_COLORS["C→C"], TOP_K_PER_PAIR, dashed=True)
draw_group(ax1, CM, pos_c, pos_m, PAIR_COLORS["C→M"], TOP_K_PER_PAIR)

def draw_nodes(ax, pos_arr, imp_arr, color):
    n = len(pos_arr)
    r = max(0.005, min(0.016, 0.50 / n))
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["#dddddd", color])
    for i, (px, py) in enumerate(pos_arr):
        imp = float(np.clip(imp_arr[i], 0, 1))
        ax.add_patch(plt.Circle((px, py), r,
                                color=cmap(0.3 + 0.7 * imp),
                                zorder=3, lw=0.5, ec="white"))

draw_nodes(ax1, pos_s, imp_sensory, COLORS_NODE["sensory"])
draw_nodes(ax1, pos_i, imp_inter,   COLORS_NODE["inter"])
draw_nodes(ax1, pos_c, imp_command, COLORS_NODE["command"])
draw_nodes(ax1, pos_m, imp_motor,   COLORS_NODE["motor"])

for name, lx, n in [("Sensory", LAYER_X[0], SENSORY_N),
                     ("Inter",   LAYER_X[1], INTER_N),
                     ("Command", LAYER_X[2], COMMAND_N),
                     ("Motor",   LAYER_X[3], MOTOR_N)]:
    ax1.text(lx, 0.98, name,     ha="center", va="center",
             fontsize=10, fontweight="bold", color=COLORS_NODE[name.lower()])
    ax1.text(lx, 0.02, f"({n})", ha="center", va="center",
             fontsize=8, color=COLORS_NODE[name.lower()])

for i in range(3):
    ax1.annotate("",
        xy=(LAYER_X[i+1]-0.025, 0.5), xytext=(LAYER_X[i]+0.025, 0.5),
        arrowprops=dict(arrowstyle="-|>", color="#90A4AE",
                        lw=1.0, mutation_scale=10), zorder=0)

line_handles = [
    mlines.Line2D([], [], color=PAIR_COLORS["S→I"], lw=1.8, label="S→I  Sensory→Inter"),
    mlines.Line2D([], [], color=PAIR_COLORS["I→C"], lw=1.8, label="I→C  Inter→Command"),
    mlines.Line2D([], [], color=PAIR_COLORS["C→C"], lw=1.8, ls="--", label="C→C  Command (recurrent)"),
    mlines.Line2D([], [], color=PAIR_COLORS["C→M"], lw=1.8, label="C→M  Command→Motor"),
]
ax1.legend(handles=line_handles, loc="upper right",
           fontsize=8, framealpha=0.85, title="Connection type")

sm = plt.cm.ScalarMappable(
    cmap=mcolors.LinearSegmentedColormap.from_list("", ["#eeeeee", "#333333"]),
    norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax1, fraction=0.025, pad=0.01, location="right")
cbar.set_label("Relative trained weight magnitude", fontsize=8)

# ── 右图 ──────────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
ax2.axis("off"); ax2.set_facecolor("#F8F8F8")
ax2.set_title("CfC-NCP Full Model Architecture", fontsize=11, pad=10)

BCOL = ["#CFD8DC", "#B3E5FC", "#E1BEE7", "#C8E6C9", "#FFF9C4"]
blocks = [
    (1.0, 7.8, 8.0, 1.0, BCOL[0], "Input ECG\n(B, T=1000, 12)"),
    (1.5, 5.6, 7.0, 1.7, BCOL[2],
     "STFT Preprocessing\n"
     "torch.stft(n_fft=64, hop=16, hann_window)\n"
     "12 leads x 33 freq bins -> (B, 396, T'~63)"),
    (1.5, 3.7, 7.0, 1.4, BCOL[1],
     "CNN Frontend\nConv1d(396->32, k=5, s=1) + BN + GELU\n"
     "Conv1d(32->64,  k=5, s=1) + BN + GELU\n-> (B, T'~63, 64)"),
    (1.5, 1.9, 7.0, 1.4, BCOL[3],
     f"CfC-NCP (3 wired layers)\nSensory=64 -> Inter={INTER_N} -> Command={COMMAND_N} -> Motor={MOTOR_N}\n"
     f"Attention pooling -> (B, {MOTOR_N})"),
    (1.5, 0.8, 7.0, 0.7, BCOL[4],
     f"FC Head:  Linear({MOTOR_N} -> {N_CLASSES})  +  BCEWithLogitsLoss"),
]
for (x, y, w, h, color, label) in blocks:
    ax2.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.1", facecolor=color, edgecolor="#607D8B", linewidth=1.5))
    ax2.text(x+w/2, y+h/2, label, ha="center", va="center",
             fontsize=8.5, fontfamily="monospace")

for y_tail, y_head in [(7.8, 7.3), (5.6, 5.1), (3.7, 3.2), (1.9, 1.5)]:
    ax2.annotate("", xy=(5, y_head), xytext=(5, y_tail),
                 arrowprops=dict(arrowstyle="-|>", color="#37474F", lw=1.5))

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\n[Saved] {OUT_FILE}")