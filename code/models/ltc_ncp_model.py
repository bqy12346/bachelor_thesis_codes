import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ncps.wirings import NCP
from ncps.torch import LTC


# ──────────────────────────────────────────────────────────────────────────────
# Neural network  (same style as ltc.py / cfc.py)
# ──────────────────────────────────────────────────────────────────────────────

class NCPNet(nn.Module):
    """LTC-NCP model for multi-label ECG classification.

    Input:  (batch, time=1000, channels=12)
    Output: (batch, n_classes)   raw logits
    """

    def __init__(self, n_classes: int, motor_neurons = 32, mixed_memory = False):

        super(NCPNet, self).__init__()

        wiring = NCP(                   # equal with wiring parameters in ltc.py 
            inter_neurons=16,
            command_neurons=8,
            motor_neurons=motor_neurons,
            sensory_fanout=8,
            inter_fanout=4,
            recurrent_command_synapses=4,
            motor_fanin=4,
        )

        self.rnn = LTC(                 # class LTC in ltc.py
            input_size=12,           # 12-lead ECG
            units=wiring,
            batch_first=True,
            mixed_memory=mixed_memory,
            input_mapping="affine",
            output_mapping="affine",
            ode_unfolds=1,
            epsilon=1e-8,
            implicit_param_constraints=True,
        )

        self.fc = nn.Linear(self.rnn.output_size, n_classes)        # Map the features into scores for five categories

    def forward(self, x: torch.Tensor):
        # x: (B, T, 12)
        out, _ = self.rnn(x)     # out: (B, T, motor_neurons)
        out = out[:, -1, :]      # take last time step → (B, motor_neurons)
        return self.fc(out)      # (B, n_classes)


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
            n_classes=n_classes, 
            motor_neurons=self.motor_neurons,
            mixed_memory=self.mixed_memory,).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        X_tr = torch.tensor(X_train, dtype=torch.float32)
        y_tr = torch.tensor(y_train, dtype=torch.float32)
        X_vl = torch.tensor(X_val,   dtype=torch.float32).to(self.device)
        y_vl = torch.tensor(y_val,   dtype=torch.float32).to(self.device)

        loader = DataLoader(
            TensorDataset(X_tr, y_tr),
            batch_size=self.batch_size,
            shuffle=True,
        )

        # [1cycle newly added] create scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr          = self.lr,
            steps_per_epoch = len(loader),
            epochs          = self.epochs,
        )

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                scheduler.step()   # [1cycle newly added]

            self.model.eval()
            with torch.no_grad():
                val_loss = criterion(self.model(X_vl), y_vl).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{self.epochs}  val_loss={val_loss:.4f}")

        self.model.load_state_dict(best_state)
        print(f"Done. Best val_loss={best_val_loss:.4f}")

        # preserve the best model state
        import os
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../saved_models")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(best_state, os.path.join(save_dir, "ltc_ncp.pt"))
        print(f"Model saved to {save_dir}/ltc_ncp.pt")

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
