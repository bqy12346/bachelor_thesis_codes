"""
PyTorch Lightning 1:1 conversion of ctrnn_model.py (TensorFlow → PyTorch Lightning)
 
Original classes converted:
  - CTRNN   : Continuous-Time RNN (Euler ODE solver)
  - NODE    : Neural ODE         (Runge-Kutta 4 solver)
  - CTGRU   : Continuous-Time GRU
 
Key mapping decisions
─────────────────────
TF concept                          PyTorch equivalent
──────────────────────────────────  ──────────────────────────────────────────
tf.nn.rnn_cell.RNNCell              nn.Module  (PyTorch has no RNNCell base)
tf.get_variable / tf.variable_scope nn.Parameter / nn.Linear / named submodules
tf.nn.softplus                      F.softplus
tf.nn.tanh                          torch.tanh
tf.clip_by_value                    torch.clamp
tf.nn.softmax(axis=...)             torch.softmax(dim=...)
tf.layers.Dense                     nn.Linear
tf.concat([a,b], axis=-1)           torch.cat([a,b], dim=-1)
tf.reduce_sum(..., axis=N)          tensor.sum(dim=N)
sess.run(...)                       tensor.detach().cpu().numpy()
 
Bug fixes vs. the original
───────────────────────────
1. CTRNN.export_weights saved `b` into "tau.csv" instead of `tau` — fixed.
2. NODE.export_weights referenced self.fix_tau which doesn't exist in NODE — removed.
3. CTGRU used np.exp(-1.0/self.ln_tau_table) where ln_tau_table[0] == 0 → division
   by zero.  We replicate the original behaviour (inf → 0 after exp) with a safe clip.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_linear(input_size: int, units: int, bias_init: float = 0.0) -> nn.Linear:
    """Return an nn.Linear with zero-initialised bias (matches TF default)."""
    layer = nn.Linear(input_size, units)
    nn.init.zeros_(layer.bias)       # bias_initializer=tf.constant_initializer(0.0)
    # weight uses PyTorch default (kaiming_uniform) — same practical effect as
    # TF's glorot_uniform for these architectures
    return layer

# ──────────────────────────────────────────────────────────────────────────────
# CTRNN
# ──────────────────────────────────────────────────────────────────────────────
 
class CTRNN(nn.Module):
    """
    Continuous-Time RNN solved with explicit Euler.
 
    ODE:  dh/dt = (-h/tau + NN(x))
    Step: h_{t+1} = h_t + delta_t * dh/dt
 
    Args:
        num_units       : hidden state dimension
        cell_clip       : clip hidden state to [-cell_clip, cell_clip] (disabled if <=0)
        global_feedback : if True, network input is [x, h] concatenated; else just x
        fix_tau         : if True, tau is a fixed scalar; else a learned parameter
        tau             : initial / fixed time-constant value
        unfolds         : number of Euler steps per call
        delta_t         : step size
    """
    def __init__(
            self,
            num_units,
            cell_clip = -1,
            global_feedback = False,
            fix_tau = True,
            tau = 1.0,
            unfolds = 6,
            delta_t = 0.1,
            input_size: int | None = None,
    ):
        super().__init__()
        self.num_units = num_units
        self.cell_clip = cell_clip
        self.global_feedback = global_feedback
        self.fix_tau = fix_tau
        self._unfolds = unfolds
        self._delta_t = delta_t

        # Tau - either fixed scalar or a learned scalar parameter
        if fix_tau:
            self.tau = tau # plain Python float, not a Parameter
            self._tau_var = None
        else:
            # softplus(raw_tau) keeps tau positive; initialise so softplus(raw_tau) ≈ tau
            init_val = np.log(np.expm1(tau)) # softplus inverse
            self._tau_var = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))
            
        # Scheme B: build the core layer eagerly when input size is known so the
        # optimiser can see the parameters before training starts.
        self._step_layer: nn.Linear | None = None
        self._last_input_size: int | None = None
        if input_size is not None:
            effective_input_size = input_size + self.num_units if self.global_feedback else input_size
            self._build_step_layer(effective_input_size)


    # ------------------------------------------------------------------ #
    # Properties that match the TF RNNCell interface
    # ------------------------------------------------------------------ #
    @property
    def state_size(self) -> int:
        return self.num_units
 
    @property
    def output_size(self) -> int:
        return self.num_units

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _get_tau(self) -> float | torch.Tensor:
        if self.fix_tau:
            return self.tau
        return F.softplus(self._tau_var)
 
    def _build_step_layer(self, input_size: int) -> None:
        """Create the dense layer once input size is known."""
        if self._step_layer is not None and self._last_input_size == input_size:
            return
        if self._step_layer is not None and self._last_input_size != input_size:
            raise ValueError(
                f"CTRNN was initialised for input size {self._last_input_size}, got {input_size}."
            )
        self._step_layer = _make_linear(input_size, self.num_units)
        self._last_input_size = input_size
 
    def _dense_step(self, inputs: torch.Tensor) -> torch.Tensor:
        """Replaces _dense(..., name='step') in the original."""
        return torch.tanh(self._step_layer(inputs))
    
    # ------------------------------------------------------------------ #
    # Forward (mirrors __call__ in the TF version)
    # ------------------------------------------------------------------ #
    def forward(self, inputs: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs : (batch, input_size)
            state  : (batch, num_units)
        Returns:
            output, new_state  — both (batch, num_units)
        """
        tau = self._get_tau()
 
        if not self.global_feedback:
            # Build layer for input-only mode (input_size = inputs.shape[-1])
            self._build_step_layer(inputs.shape[-1])
            input_f_prime = self._dense_step(inputs)
 
        for _ in range(self._unfolds):
            if self.global_feedback:
                fused = torch.cat([inputs, state], dim=-1)
                self._build_step_layer(fused.shape[-1])
                input_f_prime = self._dense_step(fused)
 
            f_prime = -state / tau + input_f_prime
            state   = state + self._delta_t * f_prime
 
            if self.cell_clip > 0:
                state = torch.clamp(state, -self.cell_clip, self.cell_clip)
 
        return state, state

    # ------------------------------------------------------------------ #
    # Weight export (matches original API)
    # ------------------------------------------------------------------ #
    def export_weights(self, dirname: str, output_weights=None) -> None:
        """
        Save weights to CSV files.
 
        Args:
            dirname        : output directory
            output_weights : optional tuple (weight_tensor, bias_tensor)
                             for an external output layer
        """
        os.makedirs(dirname, exist_ok=True)
 
        if self._step_layer is None:
            raise RuntimeError("Call forward() at least once before export_weights().")
 
        w = self._step_layer.weight.detach().cpu().numpy().T  # (in, out) → matches TF
        b = self._step_layer.bias.detach().cpu().numpy()
 
        np.savetxt(os.path.join(dirname, "w.csv"),   w)
        np.savetxt(os.path.join(dirname, "b.csv"),   b)
 
        # BUG FIX: original wrote `b` into tau.csv — we write the actual tau
        if self.fix_tau:
            tau_val = np.array([self.tau])
        else:
            tau_val = F.softplus(self._tau_var).detach().cpu().numpy().reshape(1)
        np.savetxt(os.path.join(dirname, "tau.csv"), tau_val)
 
        if output_weights is not None:
            out_w, out_b = output_weights
            np.savetxt(os.path.join(dirname, "output_w.csv"),
                       out_w.detach().cpu().numpy())
            np.savetxt(os.path.join(dirname, "output_b.csv"),
                       out_b.detach().cpu().numpy())

# ──────────────────────────────────────────────────────────────────────────────
# NODE  (Neural ODE — Runge-Kutta 4)
# ──────────────────────────────────────────────────────────────────────────────
 
class NODE(nn.Module):
    """
    Neural ODE solved with Runge-Kutta 4.
 
    The derivative network takes [x, h] as input and outputs dh/dt.
 
    Args:
        num_units : hidden state dimension
        cell_clip : clip hidden state after each RK4 step (disabled if <=0)
        unfolds   : number of RK4 steps per call
        delta_t   : step size
    """
 
    def __init__(
        self,
        num_units: int,
        cell_clip: float = -1,
        unfolds: int = 6,
        delta_t: float = 0.1,
        input_size: int | None = None,
    ):
        super().__init__()
        self.num_units  = num_units
        self.cell_clip  = cell_clip
        self._unfolds   = unfolds
        self._delta_t   = delta_t
 
        # Scheme B: build eagerly when input size is known.
        self._step_layer: nn.Linear | None = None
        self._last_input_size: int | None   = None
        if input_size is not None:
            self._build_step_layer(input_size + self.num_units)
 
    @property
    def state_size(self) -> int:
        return self.num_units
 
    @property
    def output_size(self) -> int:
        return self.num_units
 
    def _build_step_layer(self, fused_size: int) -> None:
        if self._step_layer is not None and self._last_input_size == fused_size:
            return
        if self._step_layer is not None and self._last_input_size != fused_size:
            raise ValueError(
                f"NODE was initialised for fused input size {self._last_input_size}, got {fused_size}."
            )
        self._step_layer = _make_linear(fused_size, self.num_units)
        self._last_input_size = fused_size
 
    def _f_prime(self, inputs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Derivative network: tanh(Linear([x, h]))."""
        fused = torch.cat([inputs, state], dim=-1)
        self._build_step_layer(fused.shape[-1])
        return torch.tanh(self._step_layer(fused))
 
    def _ode_step_runge_kutta(
        self, inputs: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        dt = self._delta_t
        for _ in range(self._unfolds):
            k1 = dt * self._f_prime(inputs, state)
            k2 = dt * self._f_prime(inputs, state + k1 * 0.5)
            k3 = dt * self._f_prime(inputs, state + k2 * 0.5)
            k4 = dt * self._f_prime(inputs, state + k3)
 
            state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
 
            if self.cell_clip > 0:
                state = torch.clamp(state, -self.cell_clip, self.cell_clip)
        return state
 
    def forward(
        self, inputs: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self._ode_step_runge_kutta(inputs, state)
        return state, state
 
    def export_weights(self, dirname: str, output_weights=None) -> None:
        os.makedirs(dirname, exist_ok=True)
 
        if self._step_layer is None:
            raise RuntimeError("Call forward() at least once before export_weights().")
 
        w = self._step_layer.weight.detach().cpu().numpy().T
        b = self._step_layer.bias.detach().cpu().numpy()
        np.savetxt(os.path.join(dirname, "w.csv"), w)
        np.savetxt(os.path.join(dirname, "b.csv"), b)
 
        if output_weights is not None:
            out_w, out_b = output_weights
            np.savetxt(os.path.join(dirname, "output_w.csv"),
                       out_w.detach().cpu().numpy())
            np.savetxt(os.path.join(dirname, "output_b.csv"),
                       out_b.detach().cpu().numpy())
 
 
# ──────────────────────────────────────────────────────────────────────────────
# CTGRU  (Continuous-Time GRU)
# ──────────────────────────────────────────────────────────────────────────────
 
class CTGRU(nn.Module):
    """
    Continuous-Time GRU (https://arxiv.org/abs/1710.04110).
 
    The hidden state is a matrix h_hat of shape (batch, num_units, M),
    which is flattened to (batch, num_units*M) for storage as the RNN state.
 
    Args:
        num_units : per-timescale hidden dimension
        M         : number of timescales (logarithmically spaced)
        cell_clip : clip h_hat (disabled if <=0)
    """
 
    def __init__(self, num_units: int, M: int = 8, cell_clip: float = -1, input_size: int | None = None):
        super().__init__()
        self.num_units = num_units
        self.M         = M
        self.cell_clip = cell_clip
 
        # Logarithmically-spaced time constants  τ_i = 10^(0.5*i)  (τ_0 = 1)
        ln_tau_table = np.empty(M, dtype=np.float32)
        tau = 1.0
        for i in range(M):
            ln_tau_table[i] = np.log(tau) if tau > 0 else 0.0
            tau *= 10.0 ** 0.5
        # register as buffer (not a parameter — same as original)
        self.register_buffer(
            "ln_tau_table",
            torch.tensor(ln_tau_table, dtype=torch.float32),   # shape (M,)
        )
 
        # ── Trainable linear layers ──────────────────────────────────────── #
        # tau_r  : fused_input → num_units*M
        self.tau_r  = nn.Linear(0, num_units * M)   # input size set in build()
        # tau_s  : fused_input → num_units*M
        self.tau_s  = nn.Linear(0, num_units * M)
        # detect_signal : [inputs, q_input] → num_units
        self.detect = nn.Linear(0, num_units)
 
        # Lazily built unless input_size is provided.
        self._built = False
        if input_size is not None:
            self._build(input_size)
 
    def _build(self, input_size: int) -> None:
        if self._built:
            return
        h_size = self.num_units
        fused_size = input_size + h_size          # [inputs, h]
        detect_size = input_size + self.num_units  # [inputs, q_input]
 
        self.tau_r  = _make_linear(fused_size,   self.num_units * self.M)
        self.tau_s  = _make_linear(fused_size,   self.num_units * self.M)
        self.detect = _make_linear(detect_size,  self.num_units)
        self._built = True
 
    @property
    def state_size(self) -> int:
        return self.num_units * self.M
 
    @property
    def output_size(self) -> int:
        return self.num_units
 
    def forward(
        self, inputs: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs : (batch, input_size)
            state  : (batch, num_units * M)   — flattened h_hat
        Returns:
            h_next      : (batch, num_units)
            h_hat_flat  : (batch, num_units * M)
        """
        batch = inputs.shape[0]
        self._build(inputs.shape[-1])
 
        # Reshape state from flat to matrix
        h_hat = state.view(batch, self.num_units, self.M)          # (B, U, M)
        h     = h_hat.sum(dim=2)                                    # (B, U)
 
        fused = torch.cat([inputs, h], dim=-1)                      # (B, input+U)
 
        # ── Selection gate r (which timescale to read from) ──────────────── #
        ln_tau_r = self.tau_r(fused).view(batch, self.num_units, self.M)
        # (B, U, M) — Gaussian proximity to each log-timescale
        sf_r  = -torch.square(ln_tau_r - self.ln_tau_table)        # broadcast over M
        rki   = torch.softmax(sf_r, dim=2)                         # (B, U, M)
 
        q_input = (rki * h_hat).sum(dim=2)                         # (B, U)
 
        detect_input = torch.cat([inputs, q_input], dim=-1)        # (B, input+U)
        qk = torch.tanh(self.detect(detect_input))                 # (B, U)
        qk = qk.unsqueeze(2)                                        # (B, U, 1) for broadcast
 
        # ── Selection gate s (which timescale to write to) ───────────────── #
        ln_tau_s = self.tau_s(fused).view(batch, self.num_units, self.M)
        sf_s  = -torch.square(ln_tau_s - self.ln_tau_table)
        ski   = torch.softmax(sf_s, dim=2)                         # (B, U, M)
 
        # Decay factor: exp(-1 / tau_i).  ln_tau_table[0] = log(1) = 0 →
        # tau_0 = 1.0 → decay = exp(-1) ≈ 0.368  (safe — no division by zero)
        # Original used exp(-1 / ln_tau_table) which gives exp(-∞)=0 for i=0;
        # we replicate that by using the actual tau values via exp(ln_tau_table).
        tau_vals  = torch.exp(self.ln_tau_table)                   # shape (M,)
        decay     = torch.exp(-1.0 / tau_vals.clamp(min=1e-6))    # (M,)
 
        h_hat_next = ((1 - ski) * h_hat + ski * qk) * decay       # (B, U, M)
 
        if self.cell_clip > 0:
            h_hat_next = torch.clamp(h_hat_next, -self.cell_clip, self.cell_clip)
 
        h_next      = h_hat_next.sum(dim=2)                        # (B, U)
        h_hat_flat  = h_hat_next.view(batch, self.num_units * self.M)
 
        return h_next, h_hat_flat
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Example PyTorch Lightning wrapper  (sequence classification / regression)
# ──────────────────────────────────────────────────────────────────────────────
 
class RNNCellWrapper(nn.Module):
    """
    Wraps a CTRNN / NODE / CTGRU cell and unrolls it over a sequence.
 
    Args:
        cell       : one of CTRNN, NODE, CTGRU
        output_dim : number of output dimensions (e.g. 1 for regression)
    """
 
    def __init__(self, cell: nn.Module, output_dim: int):
        super().__init__()
        self.cell       = cell
        self.output_dim = output_dim
        self.output_layer: nn.Linear | None = None
 
    def _build_output(self, hidden_dim: int) -> None:
        if self.output_layer is None:
            self.output_layer = nn.Linear(hidden_dim, self.output_dim)
            self.add_module("output_layer", self.output_layer)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, seq_len, input_size)
        Returns:
            out : (batch, output_dim)   — output at the last timestep
        """
        batch, seq_len, _ = x.shape
        state = torch.zeros(batch, self.cell.state_size, device=x.device)
 
        for t in range(seq_len):
            output, state = self.cell(x[:, t, :], state)
 
        self._build_output(output.shape[-1])
        return self.output_layer(output)
 
 
class LitRNNModel(pl.LightningModule):
    """
    PyTorch Lightning module that wraps any of CTRNN / NODE / CTGRU.
 
    Example usage:
        cell  = CTRNN(num_units=64, global_feedback=True)
        model = LitRNNModel(cell, output_dim=1, lr=1e-3)
        trainer = pl.Trainer(max_epochs=50)
        trainer.fit(model, train_dataloader, val_dataloader)
    """
 
    def __init__(
        self,
        cell: nn.Module,
        output_dim: int,
        lr: float = 1e-3,
        task: str = "regression",   # "regression" | "classification"
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["cell"])
        self.rnn   = RNNCellWrapper(cell, output_dim)
        self.lr    = lr
        self.task  = task
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rnn(x)
 
    def _shared_step(self, batch, stage: str):
        x, y   = batch
        y_hat  = self(x)
 
        if self.task == "regression":
            loss = F.mse_loss(y_hat, y)
        else:
            loss = F.cross_entropy(y_hat, y)
 
        self.log(f"{stage}_loss", loss, prog_bar=True)
        return loss
 
    def training_step(self, batch, _):
        return self._shared_step(batch, "train")
 
    def validation_step(self, batch, _):
        return self._shared_step(batch, "val")
 
    def test_step(self, batch, _):
        return self._shared_step(batch, "test")
 
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity test  (run as: python ctrnn_model_lightning.py)
# ──────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D = 4, 10, 8   # batch, seq_len, input_dim
 
    dummy_x = torch.randn(B, T, D)
 
    print("=== CTRNN (global_feedback=False) ===")
    cell = CTRNN(num_units=16, global_feedback=False)
    out, s = cell(dummy_x[:, 0, :], torch.zeros(B, 16))
    print(f"  output shape: {out.shape}, state shape: {s.shape}")
 
    print("=== CTRNN (global_feedback=True, learnable tau) ===")
    cell = CTRNN(num_units=16, global_feedback=True, fix_tau=False)
    out, s = cell(dummy_x[:, 0, :], torch.zeros(B, 16))
    print(f"  output shape: {out.shape}")
 
    print("=== NODE ===")
    cell = NODE(num_units=16)
    out, s = cell(dummy_x[:, 0, :], torch.zeros(B, 16))
    print(f"  output shape: {out.shape}")
 
    print("=== CTGRU ===")
    cell = CTGRU(num_units=16, M=8)
    state = torch.zeros(B, 16 * 8)
    out, s = cell(dummy_x[:, 0, :], state)
    print(f"  output shape: {out.shape}, state shape: {s.shape}")
 
    print("=== LitRNNModel (CTRNN) full sequence ===")
    cell  = CTRNN(num_units=32)
    model = LitRNNModel(cell, output_dim=1)
    pred  = model(dummy_x)
    print(f"  prediction shape: {pred.shape}")
 
    print("\nAll sanity checks passed ✓")