import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from models.ctrnn_model_lightning import CTRNN, NODE, CTGRU


class CTRNNNet(nn.Module):
    def __init__(self, model_type, n_classes, hidden_size=64):
        super().__init__()
        input_size = 12

        if model_type == "ctrnn":
            self.cell = CTRNN(hidden_size, cell_clip=-1, input_size=input_size)
        elif model_type == "node":
            self.cell = NODE(hidden_size, cell_clip=-1, input_size=input_size)
        elif model_type == "ctgru":
            self.cell = CTGRU(hidden_size, cell_clip=-1, input_size=input_size)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        state = torch.zeros(batch, self.cell.state_size, device=x.device)
        outputs = []
        for t in range(seq_len):
            out, state = self.cell(x[:, t, :], state)
            outputs.append(out)
        out = outputs[-1]
        return self.fc(out)


class CTRNNClassifier:
    def __init__(
        self,
        model_type  = "ctrnn",
        hidden_size = 64,
        epochs      = 50,
        batch_size  = 256,
        lr          = 0.001,
    ):
        self.model_type  = model_type
        self.hidden_size = hidden_size
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.model       = None
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X_train, y_train, X_val, y_val):
        n_classes  = y_train.shape[1]
        self.model = CTRNNNet(self.model_type, n_classes, self.hidden_size).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        X_tr = torch.tensor(X_train, dtype=torch.float32)
        y_tr = torch.tensor(y_train, dtype=torch.float32)
        X_vl = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_vl = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        loader = DataLoader(
            TensorDataset(X_tr, y_tr),
            batch_size  = self.batch_size,
            shuffle     = True,
            num_workers = 0,
            pin_memory  = torch.cuda.is_available(),
        )

        best_val_loss = float("inf")
        best_state    = None
        patience      = 10
        no_improve    = 0

        for epoch in tqdm(range(self.epochs), desc=self.model_type):
            self.model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_loss = criterion(self.model(X_vl), y_vl).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        self.model.load_state_dict(best_state)
        print(f"Best val_loss = {best_val_loss:.4f}")

    def predict(self, X):
        self.model.eval()
        X_t   = torch.tensor(X, dtype=torch.float32)
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_t), self.batch_size):
                xb    = X_t[i: i + self.batch_size].to(self.device)
                probs = torch.sigmoid(self.model(xb))
                preds.append(probs.cpu().numpy())
        return np.concatenate(preds, axis=0)
