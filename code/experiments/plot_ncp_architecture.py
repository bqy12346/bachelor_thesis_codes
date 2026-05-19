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
plot_cfc_ncp_wiring.py  — 按连线类型分组着色版
每对层（Sensory→Inter, Inter→Command, Command→Motor 等）独立控制线条数量和颜色
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from ncps.wirings import NCP

# ══════════════════════════════════════════════════════════════════════════════
#  ★ 修改这里 ★
PT_FILE = "/scratch2/bsc26f19/projects/bachelor_thesis_codes/code/saved_models/cfc_ncp_stft_20260519_114735.pt"
TOP_K_PER_PAIR = 40      # 每对层之间最多显示的连线数
# ══════════════════════════════════════════════════════════════════════════════

OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(PT_FILE)),
                        "cfc_ncp_wiring_weights.png")

MOTOR_NEURONS = 64
CNN_OUT_CH    = 64
N_CLASSES     = 5
INTER_N       = 32
COMMAND_N     = 16

# ── 1. 重建 wiring ────────────────────────────────────────────────────────────
wiring = NCP(
    inter_neurons=INTER_N,
    command_neurons=COMMAND_N,
    motor_neurons=MOTOR_NEURONS,
    sensory_fanout=8,
    inter_fanout=4,
    recurrent_command_synapses=4,
    motor_fanin=4,
)
wiring.build(CNN_OUT_CH)

adj         = np.array(wiring.adjacency_matrix)
sensory_adj = np.array(wiring.sensory_adjacency_matrix)
total_n     = wiring.units
SENSORY_N   = CNN_OUT_CH

# 自动判断 sensory_adj 方向
if sensory_adj.shape[0] == SENSORY_N:
    def sens_connected(s, dst): return sensory_adj[s, dst] != 0
else:
    def sens_connected(s, dst): return sensory_adj[dst, s] != 0

inter_idx   = list(range(0, INTER_N))
command_idx = list(range(INTER_N, INTER_N + COMMAND_N))
motor_idx   = list(range(INTER_N + COMMAND_N, total_n))

print(f"adj shape         : {adj.shape}")
print(f"sensory_adj shape : {sensory_adj.shape}")

# ── 2. 提取神经元重要性 ───────────────────────────────────────────────────────
state = torch.load(PT_FILE, map_location="cpu")

motor_importance = np.ones(MOTOR_NEURONS)
if "fc.weight" in state:
    w = state["fc.weight"].numpy()
    motor_importance = np.linalg.norm(w, axis=0)
    motor_importance /= motor_importance.max() + 1e-8

hidden_importance = np.ones(total_n)
for k in state.keys():
    if "output_map.weight" in k or "output_w" in k:
        w = state[k].numpy()
        hidden_importance = np.linalg.norm(w, axis=1)
        hidden_importance /= hidden_importance.max() + 1e-8
        print(f"Hidden importance ← {k}")
        break
else:
    print("output_map.weight not found → uniform")

hidden_importance[motor_idx] = motor_importance
sensory_importance = np.ones(SENSORY_N)

# ── 3. 坐标 ───────────────────────────────────────────────────────────────────
LAYER_X = [0.10, 0.35, 0.62, 0.87]

def make_pos(n, x):
    return np.array([(x, y) for y in np.linspace(0.94, 0.06, n)])

pos_s = make_pos(SENSORY_N,     LAYER_X[0])
pos_i = make_pos(INTER_N,       LAYER_X[1])
pos_c = make_pos(COMMAND_N,     LAYER_X[2])
pos_m = make_pos(MOTOR_NEURONS, LAYER_X[3])

def hid_xy(idx):
    if idx < INTER_N:
        return pos_i[idx]
    elif idx < INTER_N + COMMAND_N:
        return pos_c[idx - INTER_N]
    else:
        return pos_m[idx - INTER_N - COMMAND_N]

# ── 4. 按层对收集连线 ─────────────────────────────────────────────────────────
# 每条边 = (p_src, p_dst, weight)
pairs = {
    "S→I":  [],   # Sensory  → Inter
    "S→C":  [],   # Sensory  → Command
    "S→M":  [],   # Sensory  → Motor (理论上没有，但如果 wiring 定义了就画出来)
    "I→C":  [],   # Inter    → Command
    "I→M":  [],   # Inter    → Motor
    "C→M":  [],   # Command  → Motor
    "C→C":  [],   # Command  → Command (recurrent)
}

# sensory 连线
for dst in range(total_n):
    for s in range(SENSORY_N):
        if sens_connected(s, dst):
            w = (hidden_importance[dst] + sensory_importance[s]) / 2
            if dst < INTER_N:
                pairs["S→I"].append((pos_s[s], pos_i[dst], w))
            elif dst < INTER_N + COMMAND_N:
                pairs["S→C"].append((pos_s[s], pos_c[dst - INTER_N], w))
            else:
                pairs["S→M"].append((pos_s[s], pos_m[dst - INTER_N - COMMAND_N], w))
# hidden→hidden 连线
for dst in range(total_n):
    for src in range(total_n):
        if adj[dst, src] == 0:
            continue
        w = (hidden_importance[src] + hidden_importance[dst]) / 2
        p0 = hid_xy(src)
        p1 = hid_xy(dst)
        if src < INTER_N and INTER_N <= dst < INTER_N + COMMAND_N:
            pairs["I→C"].append((p0, p1, w))
        elif src < INTER_N and dst >= INTER_N + COMMAND_N:
            pairs["I→M"].append((p0, p1, w))
        elif INTER_N <= src < INTER_N + COMMAND_N and dst >= INTER_N + COMMAND_N:
            pairs["C→M"].append((p0, p1, w))
        elif INTER_N <= src < INTER_N + COMMAND_N and INTER_N <= dst < INTER_N + COMMAND_N:
            pairs["C→C"].append((p0, p1, w))

for k, v in pairs.items():
    print(f"  {k}: {len(v)} edges")

# ── 5. 每对层的颜色定义 ───────────────────────────────────────────────────────
# 颜色 = 目标层的颜色（深色），来源层的颜色（浅色）混合
PAIR_COLORS = {
    "S→I": "#E65100",   # 橙（inter 色）
    "S→C": "#00838F",   # 青（command 色）
    "S→M": "#AB47BC",   # 绿（motor 色）
    "I→C": "#00ACC1",   # 浅青
    "I→M": "#66BB6A",   # 浅绿
    "C→M": "#2E7D32",   # 深绿（command→motor，最重要）
    "C→C": "#006064",   # 深青（循环连接，虚线）
}
PAIR_DASHED = {"C→C"}   # 循环连接用虚线

# ── 6. 绘图 ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 9))
fig.patch.set_facecolor("#F8F8F8")

ax1 = fig.add_subplot(1, 2, 1)
ax1.set_facecolor("#F8F8F8")
ax1.axis("off")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_title(
    f"CfC-NCP Wiring  (weight-based, layer layout)\n"
    f"Sensory={SENSORY_N}  Inter={INTER_N}  Command={COMMAND_N}  Motor={MOTOR_NEURONS}",
    fontsize=11, pad=10,
)

def draw_group(ax, edges, color, top_k, dashed=False):
    if not edges:
        return
    # 按权重排序取 Top-K
    top = sorted(edges, key=lambda e: e[2], reverse=True)[:top_k]
    ls = "--" if dashed else "-"
    for (p0, p1, w) in top:
        w = float(np.clip(w, 0, 1))
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                color=color,
                alpha=0.18 + 0.60 * w,
                linewidth=0.3 + 1.8 * w,
                linestyle=ls,
                zorder=1)

# 按从左到右的层对顺序画，先画远的再画近的，避免遮挡
draw_order = ["S→I", "S→C", "S→M", "I→C", "I→M", "C→M", "C→C"]
for key in draw_order:
    draw_group(ax1, pairs[key], PAIR_COLORS[key],
               TOP_K_PER_PAIR, dashed=(key in PAIR_DASHED))

# 节点
COLORS_NODE = {
    "sensory": "#7B2D8B",
    "inter":   "#E65100",
    "command": "#00838F",
    "motor":   "#2E7D32",
}

def draw_nodes(ax, pos_arr, imp_arr, color):
    n = len(pos_arr)
    r = max(0.005, min(0.016, 0.50 / n))
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["#dddddd", color])
    for i, (px, py) in enumerate(pos_arr):
        imp = float(np.clip(imp_arr[i], 0, 1)) if i < len(imp_arr) else 0.5
        ax.add_patch(plt.Circle((px, py), r,
                                color=cmap(0.3 + 0.7 * imp),
                                zorder=3, linewidth=0.5, ec="white"))

draw_nodes(ax1, pos_s, sensory_importance,              COLORS_NODE["sensory"])
draw_nodes(ax1, pos_i, hidden_importance[inter_idx],    COLORS_NODE["inter"])
draw_nodes(ax1, pos_c, hidden_importance[command_idx],  COLORS_NODE["command"])
draw_nodes(ax1, pos_m, motor_importance,                COLORS_NODE["motor"])

# 层标签
for name, lx, n in [("Sensory", LAYER_X[0], SENSORY_N),
                     ("Inter",   LAYER_X[1], INTER_N),
                     ("Command", LAYER_X[2], COMMAND_N),
                     ("Motor",   LAYER_X[3], MOTOR_NEURONS)]:
    ax1.text(lx, 0.98, name,      ha="center", va="center",
             fontsize=10, fontweight="bold", color=COLORS_NODE[name.lower()])
    ax1.text(lx, 0.02, f"({n})",  ha="center", va="center",
             fontsize=8, color=COLORS_NODE[name.lower()])

# 数据流箭头
for i in range(3):
    ax1.annotate("",
        xy=(LAYER_X[i+1] - 0.025, 0.5),
        xytext=(LAYER_X[i] + 0.025, 0.5),
        arrowprops=dict(arrowstyle="-|>", color="#90A4AE",
                        lw=1.0, mutation_scale=10), zorder=0)

# 图例：连线类型
line_handles = [
    mlines.Line2D([], [], color=PAIR_COLORS[k],
                  linewidth=1.5,
                  linestyle="--" if k in PAIR_DASHED else "-",
                  label=k)
    for k in draw_order
    if pairs[k]
]

ax1.legend(handles=line_handles, loc="upper right",
           fontsize=8, framealpha=0.85, title="Connection type")

# 颜色条
sm = plt.cm.ScalarMappable(
    cmap=mcolors.LinearSegmentedColormap.from_list("", ["#eeeeee", "#333333"]),
    norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax1, fraction=0.025, pad=0.01, location="right")
cbar.set_label("Relative weight magnitude", fontsize=8)

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
     "12 leads × 33 freq bins → (B, 396, T'≈63)"),
    (1.5, 3.7, 7.0, 1.4, BCOL[1],
     "CNN Frontend\nConv1d(396→32, k=5, s=1)  +  BN  +  GELU\n"
     "Conv1d(32→64,  k=5, s=1)  +  BN  +  GELU\n→ (B, T'≈63, 64)"),
    (1.5, 1.9, 7.0, 1.4, BCOL[3],
     f"CfC-NCP\nSensory=64  Inter={INTER_N}  Command={COMMAND_N}  Motor={MOTOR_NEURONS}\n"
     f"Attention pooling → (B, {MOTOR_NEURONS})"),
    (1.5, 0.8, 7.0, 0.7, BCOL[4],
     f"FC Head:  Linear({MOTOR_NEURONS} → {N_CLASSES})  +  BCEWithLogitsLoss"),
]
for (x, y, w, h, color, label) in blocks:
    ax2.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.1", facecolor=color,
        edgecolor="#607D8B", linewidth=1.5))
    ax2.text(x+w/2, y+h/2, label,
             ha="center", va="center", fontsize=8.5, fontfamily="monospace")

for y_tail, y_head in [(7.8, 7.3), (5.6, 5.1), (3.7, 3.2), (1.9, 1.5)]:
    ax2.annotate("", xy=(5, y_head), xytext=(5, y_tail),
                 arrowprops=dict(arrowstyle="-|>", color="#37474F", lw=1.5))

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"\n[Saved] {OUT_FILE}")