import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ncps.wirings import NCP
from ncps.torch import CfC                   #  CfC replaces LTC


# ──────────────────────────────────────────────────────────────────────────────
# [CNN新增] 超参数说明
#
# CNN_CHANNELS : CNN 中间层通道数，控制提取的局部特征维度
# CNN_OUT_CH   : CNN 输出通道数，同时也是 CfC 的 input_size
#                原来 CfC input_size=12（原始导联数），现在变为 CNN_OUT_CH
# CNN_T_OUT    : CNN 压缩后的时间步数（原始 T=1000 → CNN_T_OUT=50）
#                计算方式：1000 // stride1 // stride2 = 1000 // 4 // 5 = 50
# ──────────────────────────────────────────────────────────────────────────────
CNN_CHANNELS = 32
CNN_OUT_CH   = 64
CNN_T_OUT    = 50    # 仅作注释说明用，实际由 stride 决定，不作为参数传入


# ──────────────────────────────────────────────────────────────────────────────
# Neural network
# ──────────────────────────────────────────────────────────────────────────────

class NCPNet(nn.Module):
    """CNN + CfC-NCP model for multi-label ECG classification.

    数据流（加入 CNN 后）：
        原始 ECG (B, T=1000, 12)
            │
            ▼  [CNN新增] Conv1d 两层，并行处理，压缩时间轴
        特征序列 (B, T=50, 64)
            │
            ▼  [不变] CfC-NCP，在压缩后的序列上做时序建模
        最后步输出 (B, motor_neurons)
            │
            ▼  [不变] 全连接分类头
        分类 logits (B, n_classes)
    """

    def __init__(self, n_classes: int, motor_neurons = 32, mixed_memory = False):
        super(NCPNet, self).__init__()

        # ── [CNN新增] 卷积前端：将原始 ECG 压缩为特征序列 ──────────────────
        # Conv1d 期望输入格式：(B, channels, T)，与 RNN 的 (B, T, channels) 相反
        # 因此在 forward() 中需要 permute 调换维度
        #
        # 第一层：kernel=5 覆盖约 50ms 的局部波形，stride=4 将 T 从 1000 压到 250
        #         12导联原始信号 → 32维局部波形特征
        # 第二层：继续压缩，T 从 250 → 50（每步约代表 200ms，约一个心跳节律单元）
        #         32维 → 64维，进一步组合特征
        # BatchNorm1d：稳定各通道的特征分布，加速收敛
        # GELU：比 ReLU 更平滑，在序列模型中效果更好
        self.cnn = nn.Sequential(
            nn.Conv1d(12, CNN_CHANNELS, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(CNN_CHANNELS),
            nn.GELU(),
            nn.Conv1d(CNN_CHANNELS, CNN_OUT_CH, kernel_size=5, stride=5, padding=2),
            nn.BatchNorm1d(CNN_OUT_CH),
            nn.GELU(),
        )
        # ── [CNN新增] 结束 ──────────────────────────────────────────────────

        wiring = NCP(
            inter_neurons=16,
            command_neurons=8,
            motor_neurons=motor_neurons,
            sensory_fanout=8,
            inter_fanout=4,
            recurrent_command_synapses=4,
            motor_fanin=4,
        )

        self.rnn = CfC(
            # [CNN调整] input_size 从 12 改为 CNN_OUT_CH(=64)
            # 原来直接接收 12 导联原始电压，现在接收 CNN 提取的 64 维特征
            input_size=CNN_OUT_CH,
            units=wiring,
            batch_first=True,
            mixed_memory=mixed_memory,
            return_sequences=True,
        )

        self.fc = nn.Linear(self.rnn.output_size, n_classes)

    def forward(self, x: torch.Tensor):
        # x: (B, T=1000, 12)

        # ── [CNN新增] 卷积前端处理 ─────────────────────────────────────────
        # Conv1d 要求 (B, channels, T)，需先把时间轴和通道轴对调
        x = x.permute(0, 2, 1)          # (B, 12, 1000)
        x = self.cnn(x)                  # (B, 64, 50)
        x = x.permute(0, 2, 1)          # (B, 50, 64)  ← 恢复为 RNN 期望的格式
        # ── [CNN新增] 结束 ──────────────────────────────────────────────────

        # [不变] CfC-NCP 时序建模，现在只需处理 50 步而非 1000 步
        out, _ = self.rnn(x)             # out: (B, 50, motor_neurons)
        out = out[:, -1, :]              # 取最后一步 → (B, motor_neurons)
        return self.fc(out)              # (B, n_classes)


# ──────────────────────────────────────────────────────────────────────────────
# fit() / predict() wrapper  (required by scp_experiment.py)
# ──────────────────────────────────────────────────────────────────────────────

class NCPClassifier:

    def __init__(
        self,
        motor_neurons = 32,
        mixed_memory = False,
        epochs = 50,
        batch_size = 32,
        lr = 0.002,
    ):
        self.motor_neurons = motor_neurons
        self.mixed_memory = mixed_memory
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X_train, y_train, X_val, y_val):
        n_classes = y_train.shape[1]
        self.model = NCPNet(
            n_classes     = n_classes,
            motor_neurons = self.motor_neurons,
            mixed_memory  = self.mixed_memory,
        ).to(self.device)

        # [不变] Adam 优化器，参数包含 CNN 和 CfC-NCP 的所有权重
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)


        criterion = nn.BCEWithLogitsLoss()

        X_tr = torch.tensor(X_train, dtype=torch.float32)
        y_tr = torch.tensor(y_train, dtype=torch.float32)
        X_vl = torch.tensor(X_val,   dtype=torch.float32).to(self.device)
        y_vl = torch.tensor(y_val,   dtype=torch.float32).to(self.device)

        # num_workers=0：避免 CUDA 初始化后 fork 子进程导致死锁
        # pin_memory：仅在 CUDA 可用时开启，加速 CPU→GPU 数据传输
        loader = DataLoader(
            TensorDataset(X_tr, y_tr),
            batch_size  = self.batch_size,
            shuffle     = True,
            num_workers = 0,
            pin_memory  = torch.cuda.is_available(),
        )

        # for 1cycle learning rate
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

        for epoch in pbar:
            self.model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                scheduler.step() # 1cycle learning rate update
                epoch_loss += loss.item()

            self.model.eval()
            with torch.no_grad():
                val_loss = criterion(self.model(X_vl), y_vl).item()

            pbar.set_postfix({
                "train": f"{epoch_loss/len(loader):.4f}",
                "val":   f"{val_loss:.4f}",
                "best":  f"{best_val_loss:.4f}",
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone()
                                for k, v in self.model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    pbar.write(f"Early stopping at epoch {epoch+1}")
                    break

        pbar.close()
        self.model.load_state_dict(best_state)
        print(f"Best val_loss = {best_val_loss:.4f}")

        # preserve the best model state
        import os
        save_dir = "/scratch2/bsc26f19/projects/ecg_ptbxl_benchmarking/code/saved_models"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(best_state, os.path.join(save_dir, "cfc_ncp.pt"))
        print(f"Model saved to {save_dir}/cfc_ncp.pt")

    def predict(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_t), self.batch_size):
                xb = X_t[i : i + self.batch_size].to(self.device)
                probs = torch.sigmoid(self.model(xb))
                preds.append(probs.cpu().numpy())
        return np.concatenate(preds, axis=0)
