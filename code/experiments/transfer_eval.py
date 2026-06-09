import os, sys, glob, pickle
import numpy as np
import torch
import scipy.io
from scipy.signal import resample
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/scratch2/bsc26f19/projects/bachelor_thesis_codes/code")

# ── 标签映射（顺序对齐mlb.classes_: CD/HYP/MI/NORM/STTC）──
SNOMED_TO_CLASS = {
    "426783006": "NORM", "427084000": "NORM",
    "164889003": "CD",   "59118001":  "CD",
    "164909002": "CD",   "164884008": "CD",
    "427172004": "CD",   "426627000": "CD",   "63593006": "CD",
    "429622005": "STTC", "164867002": "STTC", "428750005": "STTC",
    "164861001": "STTC", "164873001": "STTC", "164930006": "STTC",
    "164865005": "MI",   "413844008": "MI",
}
CLASS_ORDER = ["CD", "HYP", "MI", "NORM", "STTC"]
CLASS_IDX   = {c: i for i, c in enumerate(CLASS_ORDER)}

# ── 数据加载 ──────────────────────────────────────
def load_cpsc_record(hea_path):
    mat_path = hea_path.replace(".hea", ".mat")
    mat  = scipy.io.loadmat(mat_path)
    key  = [k for k in mat.keys() if not k.startswith("_")][0]
    sig  = mat[key].astype(np.float32)
    if sig.shape[0] == 12:
        sig = sig.T

    with open(hea_path) as f:
        lines = f.readlines()

    fs     = int(lines[0].split()[2])
    n_samp = int(lines[0].split()[3])
    dx_codes = []
    for line in lines:
        if line.startswith("# Dx:"):
            dx_codes = [c.strip() for c in
                        line.replace("# Dx:", "").strip().split(",")]
            break

    if fs != 100:
        target = int(n_samp * 100 / fs)
        sig = resample(sig, target, axis=0)

    if len(sig) >= 1000:
        sig = sig[:1000]
    else:
        sig = np.concatenate(
            [sig, np.zeros((1000 - len(sig), 12), dtype=np.float32)])

    label = np.zeros(5, dtype=np.float32)
    for code in dx_codes:
        cls = SNOMED_TO_CLASS.get(code)
        if cls:
            label[CLASS_IDX[cls]] = 1.0

    return sig, label


def load_cpsc(data_dirs):
    sigs, labels = [], []
    skipped = 0
    for d in data_dirs:
        files = sorted(glob.glob(os.path.join(d, "**/*.hea"), recursive=True))
        print(f"  {len(files)} files in {d}")
        for f in files:
            try:
                s, l = load_cpsc_record(f)
                if l.sum() == 0:
                    skipped += 1
                    continue
                sigs.append(s)
                labels.append(l)
            except:
                skipped += 1
    print(f"  Loaded {len(sigs)}, skipped {skipped}")
    return np.array(sigs, dtype=np.float32), np.array(labels, dtype=np.float32)


# ── 标准化（使用PTB-XL的scaler）────────────────────
def apply_scaler(X, scaler_path):
    scaler = pickle.load(open(scaler_path, "rb"))
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(
            scaler.transform(x.flatten()[:, np.newaxis]).reshape(x_shape)
        )
    return np.array(X_tmp, dtype=np.float32)


# ── fastai模型推理 ────────────────────────────────
def predict_fastai(model_name, model_dir, X, n_classes=5,
                   input_shape=(1000, 12), batch_size=128):
    from models.fastai_model import fastai_model
    dummy_X = [X[i] for i in range(len(X))]
    dummy_y = [np.zeros(n_classes)] * len(X)
    model = fastai_model(
        name           = model_name,
        n_classes      = n_classes,
        freq           = 100,
        outputfolder   = model_dir,
        input_shape    = input_shape,
        input_size     = 10,
        input_channels = 12,
    )
    return model.predict(dummy_X)


# ── PyTorch模型推理 ───────────────────────────────
def predict_pytorch(net_class, weights_path, X,
                    motor_neurons=64, mixed_memory=True,
                    batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net_class(n_classes=5,
                    motor_neurons=motor_neurons,
                    mixed_memory=mixed_memory).to(device)
    net.load_state_dict(torch.load(weights_path, map_location=device))
    net.eval()
    X_t   = torch.tensor(X, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            xb    = X_t[i: i + batch_size].to(device)
            probs = torch.sigmoid(net(xb))
            preds.append(probs.cpu().numpy())
    return np.concatenate(preds, axis=0)


# ── AUC计算 ───────────────────────────────────────
def compute_auc(y_true, y_pred, model_name):
    valid = [i for i in range(5)
             if len(np.unique(y_true[:, i])) > 1]
    if not valid:
        print(f"  {model_name}: no valid columns")
        return None
    auc_per = roc_auc_score(
        y_true[:, valid], y_pred[:, valid], average=None)
    macro = float(np.mean(auc_per))
    print(f"\n  {model_name}  macro={macro:.4f}")
    for i, idx in enumerate(valid):
        print(f"    {CLASS_ORDER[idx]:<6}: {auc_per[i]:.4f}")
    return macro

# ── Recall计算 ──────────────────────────────────────
def compute_recall(y_true, y_pred, model_name, threshold=0.5):
    from sklearn.metrics import recall_score
    y_pred_binary = (y_pred >= threshold).astype(int)
    valid = [i for i in range(5)
             if len(np.unique(y_true[:, i])) > 1]
    if not valid:
        print(f"  {model_name}: no valid columns for recall")
        return None
    recall = recall_score(
        y_true[:, valid], y_pred_binary[:, valid],
        average='macro', zero_division=0
    )
    print(f"  {model_name}  recall={recall:.4f}")
    return float(recall)

# ── 主流程 ────────────────────────────────────────
if __name__ == "__main__":

    BASE    = "/scratch2/bsc26f19/projects/bachelor_thesis_codes"
    OUT_DIR = f"{BASE}/code/output/exp_superdiagnostic/models"
    SCALER  = f"{BASE}/code/output/exp_superdiagnostic/data/standard_scaler.pkl"
    SAVED   = f"{BASE}/code/saved_models"

    CPSC_DIRS = [
        f"{BASE}/data/cpsc/cpsc_2018",
        f"{BASE}/data/cpsc/cpsc_2018_extra",
    ]

    print("="*55)
    print("  Loading CPSC2018...")
    print("="*55)
    X_raw, y_true = load_cpsc(CPSC_DIRS)
    print(f"  Shape: {X_raw.shape}")
    for i, c in enumerate(CLASS_ORDER):
        print(f"    {c}: {int(y_true[:,i].sum())} samples")

    print("\nApplying PTB-XL scaler...")
    X = apply_scaler(X_raw, SCALER)

    results = {}

    # ── fastai模型 ──
    for name in ["fastai_lstm", "fastai_gru",
                 "fastai_resnet1d_wang", "fastai_inception1d","fastai_xresnet1d101"]:
        mdir = f"{OUT_DIR}/{name}/"
        pth  = f"{mdir}models/{name}.pth"
        if os.path.exists(pth):
            print(f"\nPredicting with {name}...")
            try:
                y_pred = predict_fastai(name, mdir, X)
                auc = compute_auc(y_true, y_pred, name)
                recall = compute_recall(y_true, y_pred, name)
                results[name] = {"auc": auc, "recall": recall}
            except Exception as e:
                print(f"  Error: {e}")
                results[name] = {"auc": None, "recall": None}

    # ── CfC-NCP ──
    # cfc_files = sorted(glob.glob(f"{SAVED}/cfc_ncp_stft_*.pt"))
    cfc_files = sorted(glob.glob(f"{BASE}/code/output/ablation/E0_full/superdiagnostic/cfc_ncp_full_*.pt"))

    cfc_pt = cfc_files[-1] if cfc_files else None
    print(f"Using CfC weights: {cfc_pt}")
    if os.path.exists(cfc_pt):
        print(f"\nPredicting with cfc_ncp...")
        # from models.cfc_ncp_model import NCPNet
        from models.cfc_ncp_full_model import NCPNet
        y_pred = predict_pytorch(NCPNet, cfc_pt, X,
                                 motor_neurons=128, mixed_memory=True)
        auc = compute_auc(y_true, y_pred, "cfc_ncp")
        recall = compute_recall(y_true, y_pred, "cfc_ncp")
        results["cfc_ncp"] = {"auc": auc, "recall": recall}

    # # ── LTC-NCP ──
    # ltc_files = sorted(glob.glob(f"{SAVED}/ltc_ncp_stft_*.pt"))
    # ltc_pt = ltc_files[-1] if ltc_files else None
    # print(f"Using LTC weights: {ltc_pt}")
    # if os.path.exists(ltc_pt):
    #     print(f"\nPredicting with ltc_ncp...")
    #     from models.ltc_ncp_model import NCPNet
    #     y_pred = predict_pytorch(NCPNet, ltc_pt, X,
    #                              motor_neurons=64, mixed_memory=True)
    #     results["ltc_ncp"] = compute_auc(y_true, y_pred, "ltc_ncp")

    # ── 汇总 ──
    print(f"\n{'='*55}")
    print("  Transfer Learning: PTB-XL → CPSC2018")
    print(f"{'='*55}")
    print(f"  {'Model':<25} {'Transfer AUC':>10} {'Transfer Recall':>10}")
    print(f"  {'-'*47}")
    for name, metrics in results.items():
        auc_str = f"{metrics['auc']:.4f}" if metrics['auc'] is not None else "N/A"
        recall_str = f"{metrics['recall']:.4f}" if metrics['recall'] is not None else "N/A"
        print(f"  {name:<25} {auc_str:>10} {recall_str:>10}")
    print(f"{'='*60}")


    # import matplotlib
    # matplotlib.use("Agg")
    # import matplotlib.pyplot as plt

    # valid_results = {k: v for k, v in results.items() if v is not None}
    # names = list(valid_results.keys())
    # aucs  = list(valid_results.values())

    # PALETTE = ["#1565C0", "#E65100", "#2E7D32", "#6A1B9A",
    #         "#00838F", "#AD1457", "#4E342E"]

    # fig, ax = plt.subplots(figsize=(10, 5))
    # fig.patch.set_facecolor("#F5F5F5")
    # ax.set_facecolor("#FAFAFA")

    # bars = ax.bar(names, aucs,
    #             color=PALETTE[:len(names)],
    #             width=0.5, edgecolor="white", linewidth=1.5)

    # auc_min = max(0.0, min(aucs) - 0.05)
    # ax.set_ylim(auc_min, 1.0)
    # ax.set_title("Transfer AUC: PTB-XL → CPSC2018",
    #             fontsize=13, fontweight="bold")
    # ax.set_ylabel("Macro AUC")
    # ax.tick_params(axis="x", rotation=18, labelsize=9)
    # ax.grid(axis="y", linestyle="--", alpha=0.6)

    # for bar, val in zip(bars, aucs):
    #     ax.text(bar.get_x() + bar.get_width() / 2,
    #             bar.get_height() + (1.0 - auc_min) * 0.01,
    #             f"{val:.4f}", ha="center", va="bottom",
    #             fontsize=8.5, fontweight="bold")

    # plt.tight_layout()
    # out = f"{BASE}/code/output/exp_superdiagnostic/results/plots/transfer_cpsc2018.png"
    # os.makedirs(os.path.dirname(out), exist_ok=True)
    # plt.savefig(out, dpi=150, bbox_inches="tight")
    # plt.close()
    # print(f"\n[Plot] saved to {out}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

valid_results = {k: v for k, v in results.items() 
                 if v["auc"] is not None}
names   = list(valid_results.keys())
aucs    = [valid_results[n]["auc"]    for n in names]
recalls = [valid_results[n]["recall"] for n in names]

PALETTE = ["#1565C0", "#E65100", "#2E7D32", "#6A1B9A",
           "#00838F", "#AD1457", "#4E342E"]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.patch.set_facecolor("#F5F5F5")

# 左:AUC
ax = axes[0]
ax.set_facecolor("#FAFAFA")
bars = ax.bar(names, aucs, color=PALETTE[:len(names)],
              width=0.5, edgecolor="white", linewidth=1.5)
ax.set_ylim(max(0.0, min(aucs)-0.05), 1.0)
ax.set_title("Transfer AUC: PTB-XL → CPSC2018", fontsize=13, fontweight="bold")
ax.set_ylabel("Macro AUC")
ax.tick_params(axis="x", rotation=18, labelsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.6)
for bar, val in zip(bars, aucs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f"{val:.4f}", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold")

# 右:Recall
ax = axes[1]
ax.set_facecolor("#FAFAFA")
bars = ax.bar(names, recalls, color=PALETTE[:len(names)],
              width=0.5, edgecolor="white", linewidth=1.5)
ax.set_ylim(max(0.0, min(recalls)-0.05), 1.0)
ax.set_title("Transfer Recall: PTB-XL → CPSC2018", fontsize=13, fontweight="bold")
ax.set_ylabel("Macro Recall (threshold=0.5)")
ax.tick_params(axis="x", rotation=18, labelsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.6)
for bar, val in zip(bars, recalls):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f"{val:.4f}", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold")

plt.tight_layout()
out = f"{BASE}/code/output/exp_superdiagnostic/results/plots/transfer_cpsc2018.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\n[Plot] saved to {out}")

