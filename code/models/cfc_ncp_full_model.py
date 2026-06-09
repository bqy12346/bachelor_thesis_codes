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
# Hyperparameters
#
# STFT parameters:
#   N_FFT      : FFT window size → 33 frequency bins
#   HOP        : hop length → ~62 time frames (compressed from 1000)
#   STFT_BINS  : N_FFT // 2 + 1 = 33
#   STFT_CH    : first CNN input channels = 12 leads × 33 freq bins = 396
#
# CNN parameters:
#   CNN_CHANNELS : intermediate channel count
#   CNN_OUT_CH   : output channel count = CfC input_size
#
# Note: CNN uses stride=1 because STFT already compresses time axis (1000 → ~62)
# ──────────────────────────────────────────────────────────────────────────────
N_FFT        = 64
HOP          = 16
STFT_BINS    = N_FFT // 2 + 1   # 33
STFT_CH      = 12 * STFT_BINS   # 396

CNN_CHANNELS = 32
CNN_OUT_CH   = 64

MASK_RATIO = 0.2     # mask 20% of time steps
MASK_PROB  = 0.8     # apply mask to 80% of training batches


def random_mask(x: torch.Tensor,
                mask_ratio: float = MASK_RATIO,
                prob: float = MASK_PROB) -> torch.Tensor:
    """Random temporal masking augmentation, applied during training only.
    x: (B, T, C) ECG signal
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
    """Full CfC-NCP model: STFT + CNN + CfC-NCP + attention pooling + random masking.

    Data flow:
        Raw ECG (B, T=1000, 12)
            │
            ▼  [STFT] short-time Fourier transform → time-frequency features
        Frequency features (B, 396, ~62)   ← 12 leads × 33 freq bins, ~62 frames
            │
            ▼  [CNN] two conv blocks, stride=1 (time axis already compressed)
        Feature sequence (B, ~62, 64)
            │
            ▼  [CfC-NCP] temporal modelling + attention pooling
        Weighted output (B, motor_neurons)
            │
            ▼  [FC] linear classification head
        Logits (B, n_classes)
    """

    def __init__(self, n_classes: int, motor_neurons: int = 32, mixed_memory: bool = False):
        super(NCPNet, self).__init__()

        # ── STFT parameters (no learnable params, config only) ─────────────
        self.n_fft = N_FFT
        self.register_buffer("window", torch.hann_window(N_FFT))
        self.hop   = HOP

        # ── CNN frontend: receives STFT time-frequency features ─────────────
        # Input channels: STFT_CH=396 (12 leads × 33 freq bins)
        # stride=1: STFT already performed temporal compression (1000 → ~62)
        self.cnn = nn.Sequential(
            nn.Conv1d(STFT_CH,      CNN_CHANNELS, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(CNN_CHANNELS),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Conv1d(CNN_CHANNELS, CNN_OUT_CH,   kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(CNN_OUT_CH),
            nn.GELU(),
            nn.Dropout(0.4),
        )

        # ── CfC-NCP recurrent core ──────────────────────────────────────────
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
            input_size=CNN_OUT_CH,
            units=wiring,
            batch_first=True,
            mixed_memory=mixed_memory,
            return_sequences=True,
        )

        # ── Classification head + attention scoring layer ───────────────────
        self.fc      = nn.Linear(self.rnn.output_size, n_classes)
        self.attn_fc = nn.Linear(self.rnn.output_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T=1000, 12)

        # ── STFT preprocessing ─────────────────────────────────────────────
        B, T, C = x.shape

        x_flat = x.permute(0, 2, 1).reshape(B * C, T)    # (B*12, 1000)

        stft = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            return_complex=True,
        )                                                  # (B*12, 33, T')

        mag     = stft.abs()                               # magnitude spectrum only
        T_prime = mag.shape[-1]                            # ≈ 62

        # reshape: (B*12, 33, T') → (B, 12*33, T') = (B, 396, T')
        mag = mag.reshape(B, C, STFT_BINS, T_prime)       # (B, 12, 33, T')
        mag = mag.reshape(B, C * STFT_BINS, T_prime)      # (B, 396, T')

        # ── CNN frontend ────────────────────────────────────────────────────
        x = self.cnn(mag)                                  # (B, 64, T')
        x = x.permute(0, 2, 1)                            # (B, T', 64)

        # ── CfC-NCP + temporal attention pooling ────────────────────────────
        out, _ = self.rnn(x)                               # (B, T', motor_neurons)
        attn   = torch.softmax(self.attn_fc(out), dim=1)  # (B, T', 1)
        out    = (out * attn).sum(dim=1)                   # (B, motor_neurons)

        return self.fc(out)                                # (B, n_classes)


# ──────────────────────────────────────────────────────────────────────────────
# fit() / predict() wrapper
# ──────────────────────────────────────────────────────────────────────────────

class NCPClassifier:

    def __init__(
        self,
        motor_neurons: int   = 32,
        mixed_memory: bool   = False,
        epochs: int          = 50,
        batch_size: int      = 256,
        lr: float            = 0.002,
        task: str            = "",
    ):
        self.task          = task
        self.motor_neurons = motor_neurons
        self.mixed_memory  = mixed_memory
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.lr            = lr
        self.model         = None
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X_train, y_train, X_val, y_val):
        n_classes  = y_train.shape[1]
        self.model = NCPNet(
            n_classes     = n_classes,
            motor_neurons = self.motor_neurons,
            mixed_memory  = self.mixed_memory,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )

        steps_per_epoch = int(np.ceil(len(X_train) / self.batch_size))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr          = self.lr,
            steps_per_epoch = steps_per_epoch,
            epochs          = self.epochs,
        )

        # class-weighted BCE loss
        y_tr = torch.tensor(y_train, dtype=torch.float32)
        pos_weight = (y_tr.shape[0] - y_tr.sum(dim=0)) / (y_tr.sum(dim=0) + 1e-6)
        pos_weight = torch.clamp(pos_weight, max=10.0).to(self.device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        X_tr = torch.tensor(X_train, dtype=torch.float32)
        X_vl = torch.tensor(X_val,   dtype=torch.float32).to(self.device)
        y_vl = torch.tensor(y_val,   dtype=torch.float32).to(self.device)

        loader = DataLoader(
            TensorDataset(X_tr, y_tr),
            batch_size  = self.batch_size,
            shuffle     = True,
            num_workers = 0,
            pin_memory  = torch.cuda.is_available(),
        )

        best_val_auc = -1.0
        best_state   = None

        pbar = tqdm(range(self.epochs), desc="CfC-NCP (full)", unit="epoch",
                    dynamic_ncols=True)

        for epoch in pbar:
            # ── training ───────────────────────────────────────────────────
            self.model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                xb = random_mask(xb)          # random masking augmentation
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                scheduler.step()

            # ── validation ─────────────────────────────────────────────────
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_vl)
                val_probs  = torch.sigmoid(val_logits).cpu().numpy()

            valid_cols = [i for i in range(y_val.shape[1])
                          if len(np.unique(y_val[:, i])) > 1]
            val_auc = float(np.mean(
                roc_auc_score(y_val[:, valid_cols], val_probs[:, valid_cols],
                              average=None)
            ))

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state   = {k: v.cpu().clone()
                                for k, v in self.model.state_dict().items()}

            pbar.set_postfix(val_auc=f"{val_auc:.4f}", best=f"{best_val_auc:.4f}")

        self.model.load_state_dict(best_state)

        # ── save checkpoint ────────────────────────────────────────────────
        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../output/ablation/E0_full", self.task
        )
        os.makedirs(save_dir, exist_ok=True)
        filename = f"cfc_ncp_full_{ts}.pt"
        torch.save(best_state, os.path.join(save_dir, filename))
        print(f"\nBest val AUC = {best_val_auc:.4f}  |  saved → {filename}")

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
