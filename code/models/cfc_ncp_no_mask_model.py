
import os
import datetime
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score


from ncps.wirings import NCP
from ncps.torch import CfC


# ──────────────────────────────────────────────────────────────────────────────
# 超参数说明
#
# STFT 参数：
#   N_FFT      : FFT 窗口大小，决定频率分辨率。64 → 33 个频率 bins
#   HOP        : 帧移，决定时间分辨率。16 → 约 62 个时间帧（从 1000 步压缩而来）
#   STFT_BINS  : 频率 bins 数 = N_FFT // 2 + 1 = 33
#   STFT_CH    : CNN 第一层输入通道 = 12 导联 × 33 频率 bins = 396
#
# CNN 参数：
#   CNN_CHANNELS : CNN 中间层通道数
#   CNN_OUT_CH   : CNN 输出通道数，同时也是 CfC 的 input_size
#
# 注意：加入 STFT 后 CNN stride 改为 1×1
#       因为 STFT 已将时间轴从 1000 压缩到 ~62，不需要 CNN 再做额外压缩
# ──────────────────────────────────────────────────────────────────────────────
N_FFT        = 64
HOP          = 16
STFT_BINS    = N_FFT // 2 + 1   # 33
STFT_CH      = 12 * STFT_BINS   # 396

CNN_CHANNELS = 32
CNN_OUT_CH   = 64

MASK_RATIO = 0.2     # 每次掩盖 20% 的时间片段
MASK_PROB  = 0.8     # 80% 的训练 batch 应用掩码

def random_mask(x: torch.Tensor,
                mask_ratio: float = MASK_RATIO,
                prob: float = MASK_PROB) -> torch.Tensor:
    """随机掩码数据增强，仅在训练时调用。
    x: (B, T, C) ECG 信号
    """
    if torch.rand(1).item() > prob:
        return x

    B, T, C = x.shape
    n_mask = max(1, int(T * mask_ratio))
    starts = torch.randint(0, T - n_mask + 1, (B,), device=x.device)

    mask = torch.ones_like(x)
    for b in range(B):
        s = int(starts[b].item())
        mask[b, s : s + n_mask, :] = 0.0
    return x * mask


# ──────────────────────────────────────────────────────────────────────────────
# Neural network
# ──────────────────────────────────────────────────────────────────────────────

class NCPNet(nn.Module):
    """STFT + CNN + CfC-NCP model for multi-label ECG classification.

    数据流：
        原始 ECG (B, T=1000, 12)
            │
            ▼  [STFT] 短时傅里叶变换，转为时频特征
        频域特征 (B, 396, ~62)     ← 12导联 × 33频率bins，~62个时间帧
            │
            ▼  [CNN] 两层卷积，提取局部时频模式（stride=1，不再压缩时间轴）
        特征序列 (B, ~62, 64)
            │
            ▼  [CfC-NCP] 时序建模 + 注意力池化
        加权输出 (B, motor_neurons)
            │
            ▼  [FC] 全连接分类头
        分类 logits (B, n_classes)
    """

    def __init__(self, n_classes: int, motor_neurons: int = 32, mixed_memory: bool = False):
        super(NCPNet, self).__init__()

        # ── STFT 参数（无可学习参数，仅存储配置）──────────────────────────
        self.n_fft = N_FFT
        self.register_buffer("window", torch.hann_window(N_FFT))
        self.hop   = HOP

        # ── CNN 前端：接收 STFT 输出的时频特征 ────────────────────────────
        # 输入通道从原来的 12（原始导联）改为 STFT_CH=396（12导联×33频率bins）
        # stride 改为 1，因为 STFT 已经完成时间轴压缩（1000 → ~62）
        self.cnn = nn.Sequential(
            nn.Conv1d(STFT_CH,      CNN_CHANNELS, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(CNN_CHANNELS),
            nn.GELU(),
            nn.Dropout(0.4),  # [新增] Dropout 正则化，减少过拟合
            nn.Conv1d(CNN_CHANNELS, CNN_OUT_CH,   kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(CNN_OUT_CH),
            nn.GELU(),
            nn.Dropout(0.4),  # [新增] 第二层后也加 Dropout，进一步增强正则化
        )

        # ── CfC-NCP 时序模型 ───────────────────────────────────────────────
        wiring = NCP(
            inter_neurons=32,
            command_neurons=16,
            motor_neurons=motor_neurons,
            sensory_fanout=8,
            inter_fanout=4,
            recurrent_command_synapses=4,
            motor_fanin=4,
        )

        self.rnn = CfC(
            input_size=CNN_OUT_CH,       # 64，与 CNN 输出通道一致
            units=wiring,
            batch_first=True,
            mixed_memory=mixed_memory,
            return_sequences=True,
        )

        # ── 分类头 + 注意力评分层 ──────────────────────────────────────────
        self.fc      = nn.Linear(self.rnn.output_size, n_classes)
        self.attn_fc = nn.Linear(self.rnn.output_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T=1000, 12)

        # ── STFT 预处理 ────────────────────────────────────────────────────
        B, T, C = x.shape

        # 将 12 导联分别展平，方便批量计算 STFT
        x_flat = x.permute(0, 2, 1).reshape(B * C, T)    # (B*12, 1000)

        stft = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            return_complex=True,
        )                                                  # (B*12, 33, T')

        mag     = stft.abs()                               # 取幅度谱，忽略相位
        T_prime = mag.shape[-1]                            # ≈ 62

        # 将 12 导联的频率特征拼接为单一通道维度，供 CNN 处理
        mag = mag.reshape(B, C, STFT_BINS, T_prime)       # (B, 12, 33, T')
        x   = mag.reshape(B, STFT_CH, T_prime)            # (B, 396, T')
        # ── STFT 结束 ──────────────────────────────────────────────────────

        # ── CNN 前端 ───────────────────────────────────────────────────────
        x = self.cnn(x)             # (B, 64, T')
        x = x.permute(0, 2, 1)     # (B, T', 64)  ← 恢复为 RNN 期望的格式
        # ── CNN 结束 ───────────────────────────────────────────────────────

        # ── CfC-NCP + 时间注意力池化 ───────────────────────────────────────
        out, _ = self.rnn(x)                               # (B, T', motor_neurons)
        attn   = torch.softmax(self.attn_fc(out), dim=1)  # (B, T', 1)
        out    = (out * attn).sum(dim=1)                   # (B, motor_neurons)
        # ── RNN 结束 ───────────────────────────────────────────────────────

        return self.fc(out)                                # (B, n_classes)


# ──────────────────────────────────────────────────────────────────────────────
# AUC helper
# ──────────────────────────────────────────────────────────────────────────────
def _macro_auc(y_true: np.ndarray, y_score: np.ndarray):
    """安全计算 macro AUC，跳过只含单一类别的列"""
    valid_cols = [i for i in range(y_true.shape[1])
                  if len(np.unique(y_true[:, i])) > 1]
    if not valid_cols:
        return float("nan")
    aucs = roc_auc_score(y_true[:, valid_cols], y_score[:, valid_cols], average=None)
    return float(np.mean(aucs))


# ──────────────────────────────────────────────────────────────────────────────
# fit() / predict() wrapper  (required by scp_experiment.py)
# ──────────────────────────────────────────────────────────────────────────────

class NCPClassifier:

    def __init__(
        self,
        motor_neurons: int   = 32,
        mixed_memory:  bool  = False,
        epochs:        int   = 50,
        batch_size:    int   = 32,
        lr:            float = 0.002,
        task:          str   = "",      # add special folder to each task
    ):
        self.task = task    # add special folder to each task
        self.motor_neurons = motor_neurons
        self.mixed_memory  = mixed_memory
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.lr            = lr
        self.model         = None
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # ── [新增] 训练历史，训练完后可用 plot_training_history() 画图 ──
        self.history = {
            "train_loss": [],   # 每个 epoch 一个值
            "val_loss":   [],   # 每个 epoch 一个值
            "val_auc":    [],   # 每个 epoch 一个值（macro AUC on validation set）
            "lr_per_step":[],   # 每个 step 一个值，反映 OneCycle 曲线
        }


    def fit(self, X_train, y_train, X_val, y_val):
        n_classes = y_train.shape[1]
        self.model = NCPNet(
            n_classes     = n_classes,
            motor_neurons = self.motor_neurons,
            mixed_memory  = self.mixed_memory,
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        X_tr = torch.tensor(X_train, dtype=torch.float32)
        y_tr = torch.tensor(y_train, dtype=torch.float32)
        X_vl = torch.tensor(X_val,   dtype=torch.float32).to(self.device)
        y_vl = torch.tensor(y_val,   dtype=torch.float32).to(self.device)

        # 类别不平衡权重
        pos_weight = (y_tr.shape[0] - y_tr.sum(dim=0)) / (y_tr.sum(dim=0) + 1e-6)
        pos_weight = torch.clamp(pos_weight, max=10.0).to(self.device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        loader = DataLoader(
            TensorDataset(X_tr, y_tr),
            batch_size  = self.batch_size,
            shuffle     = True,
            num_workers = 0,
            pin_memory  = torch.cuda.is_available(),
        )

        # OneCycleLR：先升后降，帮助模型跳出局部最优
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr          = self.lr,
            steps_per_epoch = len(loader),
            epochs          = self.epochs,
        )

        best_val_loss = float("inf")
        best_state    = None
        patience      = self.epochs
        no_improve    = 0

        pbar = tqdm(range(self.epochs), desc="CfC-NCP", unit="epoch", dynamic_ncols=True)

        y_val_np = y_vl.cpu().numpy()

        best_val_auc = -1.0

        for epoch in pbar:
            self.model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                # xb = random_mask(xb) # [新增] 训练时随机掩码数据增强 # E3 版本不使用随机掩码增强
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
                # ── [新增] 记录每个 step 的学习率 ──
                self.history["lr_per_step"].append(optimizer.param_groups[0]["lr"])
            
            mean_train_loss = epoch_loss / len(loader)

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_vl)
                val_loss = criterion(val_logits, y_vl).item()
                val_probs = torch.sigmoid(val_logits).cpu().numpy()

            val_auc = _macro_auc(y_val_np, val_probs)
                        
            # ── [新增] 记录 epoch 级指标 ──
            self.history["train_loss"].append(mean_train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_auc"].append(val_auc)

            pbar.set_postfix({
                "train": f"{mean_train_loss:.4f}",
                "val":   f"{val_loss:.4f}",
                "valAUC": f"{val_auc:.4f}",
                "bestAUC":  f"{best_val_auc:.4f}",
            })

            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state   = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve   = 0
            else:
                no_improve += 1

            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     best_state    = {k: v.cpu().clone()
            #                      for k, v in self.model.state_dict().items()}
            #     no_improve    = 0
            # else:
            #     no_improve += 1
            #     if no_improve >= patience:
            #         pbar.write(f"Early stopping at epoch {epoch + 1}")
            #         break

        pbar.close()
        self.model.load_state_dict(best_state)
        print(f"Best val_auc = {best_val_auc:.4f}")

        # 保存最优模型
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../output/ablation/E3_no_mask", self.task)
        os.makedirs(save_dir, exist_ok=True)
        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cfc_ncp_no_mask_{ts}.pt"
        torch.save(best_state, os.path.join(save_dir, filename))
        print(f"Model saved to {save_dir}/{filename}")
        
        # ── [新增] 保存训练历史，供后续分析和曲线绘制 ──
        history_path = os.path.join(save_dir, f"cfc_ncp_no_mask_history_{ts}.npz")
        np.savez(
            history_path,
            train_loss  = np.array(self.history["train_loss"]),
            val_loss    = np.array(self.history["val_loss"]),
            val_auc     = np.array(self.history["val_auc"]),
            lr_per_step = np.array(self.history["lr_per_step"]),
        )
        print(f"History saved to {history_path}")
 
        # ── [新增] 自动画训练曲线 ──
        plot_path = os.path.join(save_dir, f"cfc_ncp_no_mask_curves_{ts}.png")
        plot_training_history(history_path, plot_path)
        print(f"Curves saved to {plot_path}")

    def predict(self, X):
        self.model.eval()
        X_t   = torch.tensor(X, dtype=torch.float32)
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_t), self.batch_size):
                xb    = X_t[i : i + self.batch_size].to(self.device)
                probs = torch.sigmoid(self.model(xb))
                preds.append(probs.cpu().numpy())
        return np.concatenate(preds, axis=0)
    

# ──────────────────────────────────────────────────────────────────────────────
# [新增] 训练曲线绘制函数
# ──────────────────────────────────────────────────────────────────────────────
def plot_training_history(history_path: str, out_path: str = None):
    """读取 .npz 历史文件并画 4 张子图：
       (1) train/val loss
       (2) val AUC
       (3) lr per step (OneCycle 曲线)
       (4) lr per epoch (各 epoch 的平均 lr，便于和 epoch 级指标对照)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    data = np.load(history_path)
    train_loss = data["train_loss"]
    val_loss   = data["val_loss"]
    val_auc    = data["val_auc"]
    lr_step    = data["lr_per_step"]
 
    n_epochs        = len(train_loss)
    steps_per_epoch = max(1, len(lr_step) // n_epochs)
    epochs          = np.arange(1, n_epochs + 1)
 
    # 每个 epoch 的 lr 取该 epoch 内的平均值
    lr_epoch = np.array([
        lr_step[i*steps_per_epoch : (i+1)*steps_per_epoch].mean()
        for i in range(n_epochs)
    ])
 
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.patch.set_facecolor("#F5F5F5")
 
    PALETTE = {"train": "#1565C0", "val": "#E65100",
               "auc":   "#2E7D32", "lr":  "#6A1B9A"}
 
    # ── 子图1：Loss ──
    ax = axes[0]
    ax.plot(epochs, train_loss, color=PALETTE["train"], lw=2, label="Train loss", marker="o", markersize=3)
    ax.plot(epochs, val_loss,   color=PALETTE["val"],   lw=2, label="Val loss",   marker="o", markersize=3)
    ax.set_title("Loss Curves", fontsize=12, fontweight="bold", color="#212121")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(frameon=True, framealpha=0.85)
    ax.grid(True, linestyle="--", alpha=0.5)
 
    # ── 子图2：Val AUC ──
    ax = axes[1]
    ax.plot(epochs, val_auc, color=PALETTE["auc"], lw=2, marker="o", markersize=3)
    best_epoch = int(np.nanargmax(val_auc)) + 1
    best_auc   = float(np.nanmax(val_auc))
    ax.axvline(best_epoch, color="#9E9E9E", ls=":", lw=1)
    ax.annotate(f"best epoch={best_epoch}\nAUC={best_auc:.4f}",
                xy=(best_epoch, best_auc),
                xytext=(8, -25), textcoords="offset points",
                fontsize=9, color="#212121",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDBDBD"))
    ax.set_title("Validation AUC (macro)", fontsize=12, fontweight="bold", color="#212121")
    ax.set_xlabel("Epoch"); ax.set_ylabel("AUC")
    ax.grid(True, linestyle="--", alpha=0.5)
 
    # ── 子图3：lr per step（OneCycle 完整曲线） ──
    ax = axes[2]
    ax.plot(np.arange(len(lr_step)), lr_step, color=PALETTE["lr"], lw=1.2)
    ax.set_title("Learning Rate (per step)", fontsize=12, fontweight="bold", color="#212121")
    ax.set_xlabel("Step"); ax.set_ylabel("lr")
    ax.grid(True, linestyle="--", alpha=0.5)
 
    # ── 子图4：lr per epoch（与 epoch 级指标对齐） ──
    ax = axes[3]
    ax.plot(epochs, lr_epoch, color=PALETTE["lr"], lw=2, marker="o", markersize=3)
    ax.set_title("Learning Rate (per epoch avg)", fontsize=12, fontweight="bold", color="#212121")
    ax.set_xlabel("Epoch"); ax.set_ylabel("lr")
    ax.grid(True, linestyle="--", alpha=0.5)
 
    for ax in axes:
        ax.set_facecolor("#FAFAFA")
        for s in ax.spines.values():
            s.set_edgecolor("#BDBDBD"); s.set_linewidth(0.8)
 
    fig.suptitle(f"CfC-NCP Training Curves  |  {os.path.basename(history_path)}",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
 
    if out_path is None:
        out_path = history_path.replace(".npz", "_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return out_path