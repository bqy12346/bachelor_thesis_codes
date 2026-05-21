import sys, os, time, datetime
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/scratch2/bsc26f19/projects/bachelor_thesis_codes/code")

from scp_experiment import SCP_Experiment
import torch.nn as nn
import numpy as np
import csv                                                  # [统计新增]
import matplotlib                                           # [统计新增]
matplotlib.use("Agg")                                       # 无显示器环境
import matplotlib.pyplot as plt                             # [统计新增]
import matplotlib.ticker as mticker                         # [统计新增]
from sklearn.metrics import roc_auc_score



# ── 日志系统 ──────────────────────────────────────
class Logger:
    def __init__(self, log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.terminal = sys.stdout
        self.log_file = open(log_path, "w", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join("./output/", "exp_superdiagnostic", "logs", f"exp_superdiagnostic_{timestamp}.log")
logger    = Logger(log_path)
sys.stdout = logger
# ─────────────────────────────────────────────────

# ── AUC 辅助函数 ──────────────────────────────────
def compute_auc(y_true, y_pred, split_name):
    valid_cols = [i for i in range(y_true.shape[1]) if len(np.unique(y_true[:, i])) > 1]
    if not valid_cols:
        print(f"  [{split_name} AUC] 无有效标签列，跳过计算")
        return None
    auc_per_class = roc_auc_score(
        y_true[:, valid_cols], y_pred[:, valid_cols], average=None)
    macro_auc = float(np.mean(auc_per_class))
    print(f"  [{split_name} AUC] macro = {macro_auc:.4f}  " f"(基于 {len(valid_cols)}/{y_true.shape[1]} 个有效类别)")
    print(f"           per-class: " + "  ".join(f"{v:.3f}" for v in auc_per_class))
    return macro_auc
# ─────────────────────────────────────────────────

# ── Recall 辅助函数 ──────────────────────────────────
def compute_recall(y_true, y_pred, split_name, threshold=0.5):
    from sklearn.metrics import recall_score
    y_pred_binary = (y_pred >= threshold).astype(int)
    valid_cols = [i for i in range(y_true.shape[1])
                  if len(np.unique(y_true[:, i])) > 1]
    if not valid_cols:
        return None
    recall = recall_score(
        y_true[:, valid_cols], y_pred_binary[:, valid_cols],
        average='macro', zero_division=0)
    print(f"  [{split_name} Recall] macro = {recall:.4f}")
    return float(recall)
# ─────────────────────────────────────────────────

# ── [统计新增] CSV 保存 ────────────────────────────
def save_results_csv(records, csv_path):
    """
    records: list of dict，每条对应一个模型的完整统计
    字段: model, total_params, trainable_params,
          train_time_s, val_auc, test_auc, test_recall, timestamp
    追加写入，支持多次实验累积对比。
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["model", "total_params", "trainable_params",
                  "train_time_s", "val_auc", "test_auc", "test_recall", "timestamp"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"\n  [CSV] is saved to: {csv_path}")
# ─────────────────────────────────────────────────

# ── [统计新增] 对比图绘制 ──────────────────────────
def plot_comparison(records, plot_dir):
    """
    绘制五张子图：训练时长 / Val AUC / Test AUC / Test Recall / Total Parameters，按模型分组。
    ensemble 不参与训练时长/参数图，但参与 AUC 图。
    保存为 comparison_<timestamp>.png
    """
    os.makedirs(plot_dir, exist_ok=True)

    non_ens   = [r for r in records if r["model"] != "ensemble"]
    all_rec   = records

    # 训练时长（ensemble 无意义，排除）
    # names_ne  = [r["model"] for r in non_ens]
    # times     = [float(r["train_time_s"]) if r["train_time_s"] != "N/A" else 0
    #              for r in non_ens]
    non_ens_valid = [r for r in non_ens if r["train_time_s"] != "N/A"]
    names_ne = [r["model"] for r in non_ens_valid]
    times    = [float(r["train_time_s"]) for r in non_ens_valid]
    # AUC（含 ensemble）
    names_all = [r["model"] for r in all_rec]
    val_aucs  = [float(r["val_auc"])  if r["val_auc"]  != "N/A" else 0 for r in all_rec]
    test_aucs = [float(r["test_auc"]) if r["test_auc"] != "N/A" else 0 for r in all_rec]

    MODEL_COLORS = {
        "cfc_ncp":              "#1565C0",
        "ltc_ncp":              "#0D47A1",
        "fastai_xresnet1d101":  "#E65100",
        "fastai_inception1d":   "#EF6C00",
        "fastai_resnet1d_wang": "#FF8F00",
        "fastai_lstm":          "#2E7D32",
        "fastai_gru":           "#558B2F",
        "wavelet_rf":           "#6A1B9A",
        "ctrnn":                "#00838F",
        "node":                 "#00695C",
        "ctgru":                "#26A69A",
        "ensemble":             "#37474F",
    }
    DEFAULT_COLOR = "#9E9E9E"

    fig, axes = plt.subplots(1, 5, figsize=(27, 5.5))
    fig.patch.set_facecolor("#F5F5F5")
    for ax in axes:
        ax.set_facecolor("#FAFAFA")
        for spine in ax.spines.values():
            spine.set_edgecolor("#BDBDBD")
            spine.set_linewidth(0.8)

    def bar_chart(ax, names, values, title, ylabel, fmt=".1f", ylim=None):
        colors = [MODEL_COLORS.get(name, DEFAULT_COLOR) for name in names]
        bars = ax.bar(names, values, color=colors, width=0.5,
                      edgecolor="white", linewidth=1.5, zorder=3)
        ax.set_title(title, fontsize=12, fontweight="bold",
                     pad=10, color="#212121")
        ax.set_ylabel(ylabel, fontsize=10, color="#616161")
        ax.tick_params(axis="x", labelsize=9, rotation=18)
        ax.tick_params(axis="y", labelsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(f"%{fmt}"))
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6, zorder=0)
        if ylim:
            ax.set_ylim(*ylim)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01,
                    f"{val:{fmt}}", ha="center", va="bottom",
                    fontsize=8.5, color="#212121", fontweight="bold")

    # 子图1：训练时长（秒）
    bar_chart(axes[0], names_ne, times,
              "Training Time (s)", "Seconds", fmt=".1f")

    # 子图2: 参数量（只统计有参数数据的模型）
    non_ens_params = [r for r in non_ens if r["total_params"] != "N/A"]
    names_params   = [r["model"] for r in non_ens_params]
    params_vals    = [int(r["total_params"]) for r in non_ens_params]
    bar_chart(axes[1], names_params, params_vals, "Total Parameters", "Parameters", fmt=".0f")
   
    # 子图3：Val AUC
    auc_min = max(0.0, min(v for v in val_aucs if v > 0) - 0.06) if any(val_aucs) else 0
    bar_chart(axes[2], names_all, val_aucs,
              "Validation AUC (macro)", "AUC",
              fmt=".4f", ylim=(auc_min, 1.0))

    # 子图4：Test AUC
    auc_min2 = max(0.0, min(v for v in test_aucs if v > 0) - 0.06) if any(test_aucs) else 0
    bar_chart(axes[3], names_all, test_aucs,
              "Test AUC (macro)", "AUC",
              fmt=".4f", ylim=(auc_min2, 1.0))
    
    # 子图5：Test Recall
    recall_rec = [r for r in all_rec if r.get("test_recall", "N/A") != "N/A"]
    names_recall  = [r["model"] for r in recall_rec]
    recall_vals   = [float(r["test_recall"]) for r in recall_rec]
    recall_min    = max(0.0, min(recall_vals) - 0.06) if recall_vals else 0
    bar_chart(axes[4], names_recall, recall_vals,
          "Test Recall (macro)", "Recall",
          fmt=".4f", ylim=(recall_min, 1.0))

    fig.suptitle(f"Model Comparison  |  {timestamp}",
                 fontsize=13, fontweight="bold", color="#212121", y=1.02)
    plt.tight_layout()

    out_path = os.path.join(plot_dir, f"comparison_{timestamp}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [Plot] 已保存至: {out_path}")
    return out_path
# ─────────────────────────────────────────────────

# ── 通用模型包装类 ────────────────────────────────
class ModelWrapper:
    def __init__(self, classifier, name):
        self.classifier   = classifier
        self.name         = name
        self.fit_time     = None
        self.total_params = None
        self.train_params = None

    def fit(self, X_train, y_train, X_val, y_val):
        print(f"\n>> Starting training: {self.name}")
        x_shape = np.array(X_train).shape if isinstance(X_train, list) else X_train.shape
        v_shape = np.array(X_val).shape   if isinstance(X_val,   list) else X_val.shape
        print(f"   Training set size: {x_shape},  "f"Validation set size: {v_shape}")

        t0 = time.time()
        self.classifier.fit(X_train, y_train, X_val, y_val)
        self.fit_time = time.time() - t0
        print(f"  Training time            : {self.fit_time:.1f}s")

        if hasattr(self.classifier, "model") and self.classifier.model is not None:
            model = self.classifier.model
            
            # 新增：检查是否是PyTorch模型
            if not hasattr(model, "parameters"):
                # Keras/sklearn模型，跳过参数统计
                self.total_params = None
                self.train_params = None
            else:
                self.total_params = sum(p.numel() for p in model.parameters())
                self.train_params = sum(p.numel() for p in model.parameters()
                                        if p.requires_grad)
        elif hasattr(self.classifier, "_cached_total_params"):
            self.total_params = self.classifier._cached_total_params
            self.train_params = self.classifier._cached_trainable_params
        else:
            self.total_params = None
            self.train_params = None  

        if self.total_params:
            print(f"\n{'─'*45}")
            print(f"  Total model parameters   : {self.total_params:,}")
            print(f"  Trainable parameters     : {self.train_params:,}")
            print(f"{'─'*45}")

    def predict(self, X):
        return self.classifier.predict(X)
# ─────────────────────────────────────────────────

DATAFOLDER   = "/scratch2/bsc26f19/projects/bachelor_thesis_codes/data/ptbxl/"
OUTPUTFOLDER = "./output/"

models = [

    {
        "modelname":  "cfc_ncp",
        "modeltype":  "CFC_NCP",
        "parameters": {
            "motor_neurons": 64,
            "mixed_memory":  True,
            "epochs":        50,
            "batch_size":    256,
            "lr":            0.002,
        }
    },

    # {
    #     "modelname": "ltc_ncp",
    #     "modeltype": "LTC_NCP",
    #     "parameters": {
    #         "motor_neurons": 64,
    #         "mixed_memory":  True,
    #         "epochs":        50,
    #         "batch_size":    32,
    #         "lr":            0.002,
    #     }
    # },
    
    {
        "modelname": "fastai_xresnet1d101",
        "modeltype": "FASTAI",
        "parameters": {
            "epochs": 50,
            "lr":     0.001,
        }
    },

    {
        "modelname": "fastai_inception1d",
        "modeltype": "FASTAI",
        "parameters": {
            "epochs": 50,
            "lr":     0.001,
        }
    },

    {
        "modelname": "fastai_lstm",
        "modeltype": "FASTAI",
        "parameters":{
            "epochs": 50,
            "lr": 0.001,
        }
    },

    {
        "modelname": "fastai_gru",
        "modeltype": "FASTAI",
        "parameters":{
            "epochs": 50,
            "lr": 0.001,
        }
    },

    {
        "modelname": "fastai_resnet1d_wang",
        "modeltype": "FASTAI",
        "parameters":{
            "epochs": 50,
            "lr": 0.001,
        }
    },

    {
        "modelname": "wavelet_rf",
        "modeltype": "WAVELET",
        "parameters": {
            "classifier": "NN",   # can be choosen from RF / LR / NN
        }
    },

    {
        "modelname": "ctrnn",
        "modeltype": "CTRNN_FAMILY",
        "parameters": { "model_type": "ctrnn", "hidden_size": 64, "epochs": 50, "lr": 0.001 }
    },
    # {
    #     "modelname": "node",
    #     "modeltype": "CTRNN_FAMILY",
    #     "parameters": { "model_type": "node", "hidden_size": 64, "epochs": 50, "lr": 0.001 }
    # },
    # {
    #     "modelname": "ctgru",
    #     "modeltype": "CTRNN_FAMILY",
    #     "parameters": { "model_type": "ctgru", "hidden_size": 64, "epochs": 50, "lr": 0.001 }
    # },

    
]

exp = SCP_Experiment(
    experiment_name    = "exp_superdiagnostic",
    task               = "superdiagnostic",
    datafolder         = DATAFOLDER,
    outputfolder       = OUTPUTFOLDER,
    models             = models,
    sampling_frequency = 100,
)

# ── patched perform ───────────────────────────────
auc_summary  = {}
stat_records = []   # [统计新增] 收集所有模型的完整统计行

def patched_perform(self):
    for model_description in self.models:
        modelname   = model_description['modelname']
        modeltype   = model_description['modeltype']
        modelparams = model_description['parameters']

        mpath = (self.outputfolder + self.experiment_name
                 + '/models/' + modelname + '/')
        os.makedirs(mpath, exist_ok=True)
        os.makedirs(mpath + 'results/', exist_ok=True)

        if modeltype == "CFC_NCP":
            from models.cfc_ncp_model import NCPClassifier
            raw_model = NCPClassifier(**modelparams)
        elif modeltype == "LTC_NCP":
            from models.ltc_ncp_model import NCPClassifier
            raw_model = NCPClassifier(**modelparams)
        elif modeltype == "CTRNN_FAMILY":
            from models.ctrnn_model import CTRNNClassifier
            raw_model = CTRNNClassifier(**modelparams)
        elif modeltype == "FASTAI":
            from models.fastai_model import fastai_model
            from models.rnn1d import RNN1d                          # ← 加这行
            from models.resnet1d import resnet1d_wang               # ← 加这行
            from models.inception1d import inception1d  
            from models.xresnet1d import xresnet1d101
 
            n_classes    = self.y_train.shape[1]
            input_shape  = self.X_train[0].shape   # (1000, 12)
            raw_model = fastai_model(
                name         = modelname,
                n_classes    = n_classes,
                freq         = 100,
                outputfolder = mpath,
                input_shape  = input_shape,
                epochs       = modelparams.get("epochs", 50),
                lr           = modelparams.get("lr", 1e-3),
                input_size   = 10,     # 10s window
                input_channels = 12,   # 12-lead ECG
            )

            # [新增] 训练前先算参数量，因为训练后模型不保存在内存中
            # [新增] 直接实例化对应的PyTorch模型来统计参数
            # 不需要数据集，只需要模型结构
            try:
                if modelname.startswith("fastai_lstm_bidir"):
                    _m = RNN1d(input_channels=12, num_classes=n_classes, lstm=True,  bidirectional=True)
                elif modelname.startswith("fastai_gru_bidir"):
                    _m = RNN1d(input_channels=12, num_classes=n_classes, lstm=False, bidirectional=True)
                elif modelname.startswith("fastai_lstm"):
                    _m = RNN1d(input_channels=12, num_classes=n_classes, lstm=True,  bidirectional=False)
                elif modelname.startswith("fastai_gru"):
                    _m = RNN1d(input_channels=12, num_classes=n_classes, lstm=False, bidirectional=False)
                elif modelname.startswith("fastai_resnet1d_wang"):
                    _m = resnet1d_wang(num_classes=n_classes, input_channels=12)
                elif modelname.startswith("fastai_inception1d"):
                    _m = inception1d(num_classes=n_classes, input_channels=12)
                elif modelname.startswith("fastai_xresnet1d101"):
                    _m = xresnet1d101(num_classes=n_classes, input_channels=12)
                else:
                    _m = None

                if _m is not None:
                    raw_model._cached_total_params    = sum(p.numel() for p in _m.parameters())
                    raw_model._cached_trainable_params = sum(p.numel() for p in _m.parameters()
                                                            if p.requires_grad)
                    print(f"  [参数统计] {modelname}: {raw_model._cached_total_params:,}")
                    del _m
                else:
                    raw_model._cached_total_params    = None
                    raw_model._cached_trainable_params = None
            except Exception as e:
                print(f"  [参数统计] 获取失败: {e}")
                raw_model._cached_total_params    = None
                raw_model._cached_trainable_params = None

        elif modeltype == "WAVELET":
            from models.wavelet import WaveletModel
            n_classes   = self.y_train.shape[1]
            input_shape = self.X_train[0].shape
            raw_model = WaveletModel(
                name        = modelname,
                n_classes   = n_classes,
                freq        = 100,
                outputfolder = mpath,
                input_shape = input_shape,
                classifier  = modelparams.get("classifier", "NN"),
            )

        else:
            raise ValueError(f"Unknown modeltype: {modeltype}")

        model = ModelWrapper(raw_model, modelname)
        model.fit(self.X_train, self.y_train, self.X_val, self.y_val)

        y_train_pred = model.predict(self.X_train)
        y_val_pred   = model.predict(self.X_val)
        y_test_pred  = model.predict(self.X_test)

        y_train_pred.dump(mpath + 'y_train_pred.npy')
        y_val_pred.dump(mpath   + 'y_val_pred.npy')
        y_test_pred.dump(mpath  + 'y_test_pred.npy')

        # AUC
        print(f"\n{'='*45}")
        print(f"  AUC Results: {modelname}")
        print(f"{'='*45}")
        val_auc  = compute_auc(self.y_val,  y_val_pred,  "Val")
        test_auc = compute_auc(self.y_test, y_test_pred, "Test")
        test_recall = compute_recall(self.y_test, y_test_pred, "Test")
        auc_summary[modelname] = {"val_auc": val_auc, "test_auc": test_auc, "test_recall": test_recall}

        # [统计新增] 记录该模型统计行
        stat_records.append({
            "model":            modelname,
            "total_params":     model.total_params if model.total_params else "N/A",
            "trainable_params": model.train_params if model.train_params else "N/A",
            "train_time_s":     round(model.fit_time, 1) if model.fit_time is not None else "N/A",
            "val_auc":          round(val_auc,  4) if val_auc  is not None else "N/A",
            "test_auc":         round(test_auc, 4) if test_auc is not None else "N/A",
            "test_recall":      round(test_recall, 4) if test_recall is not None else "N/A",
            "timestamp":        timestamp,
        })

    # ensemble
    ensemblepath = (self.outputfolder + self.experiment_name
                    + '/models/ensemble/')
    os.makedirs(ensemblepath, exist_ok=True)
    os.makedirs(ensemblepath + 'results/', exist_ok=True)
    ens_parts_train, ens_parts_val, ens_parts_test = [], [], []
    for m in os.listdir(self.outputfolder + self.experiment_name + '/models/'):
        if m not in ['ensemble', 'naive']:
            mp = (self.outputfolder + self.experiment_name
                  + '/models/' + m + '/')
            ens_parts_train.append(np.load(mp + 'y_train_pred.npy', allow_pickle=True))
            ens_parts_val.append(np.load(mp   + 'y_val_pred.npy',   allow_pickle=True))
            ens_parts_test.append(np.load(mp  + 'y_test_pred.npy',  allow_pickle=True))

    ens_val_pred  = np.array(ens_parts_val).mean(axis=0)
    ens_test_pred = np.array(ens_parts_test).mean(axis=0)
    np.array(ens_parts_train).mean(axis=0).dump(ensemblepath + 'y_train_pred.npy')
    ens_test_pred.dump(ensemblepath + 'y_test_pred.npy')
    ens_val_pred.dump(ensemblepath  + 'y_val_pred.npy')

    print(f"\n{'='*45}")
    print(f"  AUC Results: ensemble")
    print(f"{'='*45}")
    ens_val_auc  = compute_auc(self.y_val,  ens_val_pred,  "Val")
    ens_test_auc = compute_auc(self.y_test, ens_test_pred, "Test")
    ens_test_recall = compute_recall(self.y_test, ens_test_pred, "Test")
    auc_summary["ensemble"] = {"val_auc": ens_val_auc, "test_auc": ens_test_auc, "test_recall": ens_test_recall}

    # [统计新增] ensemble 行
    stat_records.append({
        "model":            "ensemble",
        "total_params":     "N/A",
        "trainable_params": "N/A",
        "train_time_s":     "N/A",
        "val_auc":          round(ens_val_auc,  4) if ens_val_auc  is not None else "N/A",
        "test_auc":         round(ens_test_auc, 4) if ens_test_auc is not None else "N/A",
        "test_recall":      round(ens_test_recall, 4) if ens_test_recall is not None else "N/A",
        "timestamp":        timestamp,
    })

import types
exp.perform = types.MethodType(patched_perform, exp)

# ── 主流程 ────────────────────────────────────────
try:
    print(f"Experiment started: {timestamp}")
    print(f"Log path: {log_path}\n")
    timing = {}

    print("="*50)
    print("  Step 1: Data Preparation")
    print("="*50)

    t0 = time.time()
    exp.prepare()
    timing["prepare"] = time.time() - t0
    print(f"  Time elapsed: {timing['prepare']:.1f}s")

    print("\n" + "="*50)
    print("  Step 2: Training Models")
    print("="*50)
    t0 = time.time()
    exp.perform()
    timing["train"] = time.time() - t0

    print("\n" + "="*50)
    print("  Step 3: Evaluating Results")
    print("="*50)
    t0 = time.time()
    exp.evaluate(n_bootstraping_samples=100, n_jobs=4, bootstrap_eval=False)
    timing["evaluate"] = time.time() - t0
    print(f"  Time elapsed: {timing['evaluate']:.1f}s")

    # ── Runtime Summary ───────────────────────────
    print("\n" + "="*50)
    print("  Runtime Summary")
    print("="*50)
    total = sum(timing.values())
    for k, v in timing.items():
        print(f"  {k:<20} {v:>8.1f}s  ({v/total*100:.1f}%)")
    print(f"  {'Total':<20} {total:>8.1f}s")
    print("="*50)

    # ── AUC Summary ───────────────────────────────
    print("\n" + "="*50)
    print("  AUC Summary")
    print("="*50)
    print(f"  {'Model':<20} {'Val AUC':>10} {'Test AUC':>10} {'Test Recall':>10}")
    print(f"  {'-'*52}")
    for mname, aucs in auc_summary.items():
        val_str  = f"{aucs['val_auc']:.4f}"  if aucs['val_auc']  is not None else "   N/A"
        test_str = f"{aucs['test_auc']:.4f}" if aucs['test_auc'] is not None else "   N/A"
        recall_str = f"{aucs['test_recall']:.4f}" if aucs['test_recall'] is not None else "   N/A"
        print(f"  {mname:<20} {val_str:>10} {test_str:>10} {recall_str:>10}")
    print("="*50)

    # ── [统计新增] 保存 CSV ───────────────────────
    csv_path = os.path.join(OUTPUTFOLDER, "exp_superdiagnostic", "results", "model_stats.csv")    
    save_results_csv(stat_records, csv_path)

    # ── [统计新增] 绘制对比图 ─────────────────────
    plot_dir = os.path.join(OUTPUTFOLDER, "exp_superdiagnostic", "results", "plots")
    plot_comparison(stat_records, plot_dir)

finally:
    sys.stdout = logger.terminal
    logger.close()
    print(f"\nLog saved to: {log_path}")