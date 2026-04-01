"""
compare_models.py  —  Quorum project: complete report comparison suite

Generates EVERYTHING needed for the report in one run:

  QUANTITATIVE (report_eval/outputs/)
  ├── metrics_summary.csv          — mean Dice + IoU per model per class  ← Table 1
  ├── per_image_metrics.csv        — every image × every model
  ├── statistical_tests.csv        — paired Wilcoxon test between every model pair
  └── confusion_matrices.csv       — pixel-level confusion per model

  FIGURES (report_eval/outputs/figures/)
  ├── bar_dice.png                 — grouped bar chart (Dice, nuc + chrom)
  ├── bar_iou.png                  — grouped bar chart (IoU, nuc + chrom)
  ├── boxplot_dice_chromocenter.png— box plot: chrom Dice distribution per model
  ├── boxplot_dice_nucleoplasm.png — box plot: nuc Dice distribution per model
  ├── scatter_chrom_dice.png       — per-image chrom Dice: best model vs all others
  ├── confusion_<model>.png        — pixel confusion matrix heatmap per model
  └── comparison_cell<id>.png      — Original | GT | all models | error maps
                                     (5 cells: worst/25th/median/75th/best chrom Dice)

How to run (from project root):
    python -m report_eval.compare_models
"""

import os
import sys
import time
import yaml
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import torch
import cv2
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.abspath(os.path.join(THIS_DIR, ".."))
OUT_DIR  = os.path.join(THIS_DIR, "outputs")
FIG_DIR  = os.path.join(OUT_DIR, "figures")
sys.path.insert(0, ROOT)

from data_utils.dataset import CellDataset

# ── Visual constants ─────────────────────────────────────────────────────────
BG_COLOR    = np.array([20,  20,  20],  dtype=np.uint8)
NUC_COLOR   = np.array([38,  120, 142], dtype=np.uint8)   # teal
CHROM_COLOR = np.array([226, 72,  12],  dtype=np.uint8)   # orange
ERR_COLOR   = np.array([220, 50,  50],  dtype=np.uint8)   # red wrong pixel

MODEL_COLORS = {
    "U-Net++"   : "#2196F3",
    "U-Net"     : "#4CAF50",
    "Cellpose"  : "#FF9800",
    "DeepLabV3" : "#9C27B0",
    "DeepLabV3+": "#F44336",
}
CLASS_NAMES = ["Background", "Nucleoplasm", "Chromocenter"]



# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def load_config():
    with open(os.path.join(ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_val_dataset(cfg):
    val_ids = CellDataset.load_split_ids(os.path.join(ROOT, "val_ids.txt"))
    return CellDataset(
        root_dir        = cfg["data_root"],
        preprocess_mode = cfg.get("preprocess_mode", "basic"),
        aug_strength    = "none",
        target_size     = tuple(cfg.get("target_size", [256, 256])),
        split_ids       = val_ids,
    )


def dice_iou(pred, gt, cls):
    p = (pred == cls); g = (gt == cls)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    denom = p.sum() + g.sum()
    return (1.0 if denom == 0 else float(2.0 * inter / denom),
            1.0 if union == 0 else float(inter / union))


def confusion_matrix_pixels(pred, gt, n=3):
    cm = np.zeros((n, n), dtype=np.int64)
    for r in range(n):
        for c in range(n):
            cm[r, c] = int(np.logical_and(gt == r, pred == c).sum())
    return cm


def colorize_mask(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask == 0] = BG_COLOR
    rgb[mask == 1] = NUC_COLOR
    rgb[mask == 2] = CHROM_COLOR
    return rgb


def error_map(pred, gt):
    rgb = np.zeros((*gt.shape, 3), dtype=np.uint8)
    rgb[pred != gt] = ERR_COLOR
    return rgb


def to_uint8(img_tensor):
    arr = img_tensor.squeeze().numpy()
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def repeat_3ch(x):
    return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x


# ════════════════════════════════════════════════════════════════════════════
# TTA (U-Net++)
# ════════════════════════════════════════════════════════════════════════════

def _tta(t, m):
    if m == "orig":  return t
    if m == "hflip": return torch.flip(t, [3])
    if m == "vflip": return torch.flip(t, [2])
    return torch.rot90(t, 1, [2, 3])

def _inv(t, m):
    if m == "orig":  return t
    if m == "hflip": return torch.flip(t, [3])
    if m == "vflip": return torch.flip(t, [2])
    return torch.rot90(t, 3, [2, 3])

def predict_tta(model, img_b, device):
    probs = []
    with torch.no_grad():
        for mode in ["orig", "hflip", "vflip", "rot90"]:
            p = torch.softmax(model(_tta(img_b, mode)), dim=1)
            probs.append(_inv(p, mode)[0].cpu())
    return torch.argmax(torch.stack(probs).mean(0), 0).numpy().astype(np.uint8)


# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ════════════════════════════════════════════════════════════════════════════

def load_unetpp(device):
    try:
        import segmentation_models_pytorch as smp
        p = os.path.join(ROOT, "architecture_team_1", "unetpp",
                         "runs_unetpp", "best_unetpp.pt")
        if not os.path.exists(p):
            print(f"  [SKIP] U-Net++ not found: {p}"); return None
        m = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None,
                             in_channels=1, classes=3).to(device)
        ckpt = torch.load(p, map_location=device)
        m.load_state_dict(ckpt["model_state_dict"]); m.eval()
        print(f"  Loaded U-Net++"); return m
    except Exception as e:
        print(f"  [SKIP] U-Net++ error: {e}"); return None


def load_unet(device):
    try:
        import segmentation_models_pytorch as smp
        p = os.path.join(ROOT, "architecture_team_2", "unet",
                         "runs_unet", "best_unet.pt")
        if not os.path.exists(p):
            print(f"  [SKIP] U-Net not found: {p}"); return None
        m = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=1, classes=3).to(device)
        ckpt = torch.load(p, map_location=device)
        m.load_state_dict(ckpt["model_state_dict"]); m.eval()
        print(f"  Loaded U-Net"); return m
    except Exception as e:
        print(f"  [SKIP] U-Net error: {e}"); return None


def load_deeplabv3(device):
    try:
        from architecture_team_4.deeplabv3_model import build_deeplabv3
        p = os.path.join(ROOT, "architecture_team_4", "deeplabv3",
                         "runs_deeplabv3", "best_deeplabv3.pt")
        if not os.path.exists(p):
            print(f"  [SKIP] DeepLabV3 not found: {p}"); return None
        ckpt = torch.load(p, map_location=device)
        m = build_deeplabv3(backbone=ckpt.get("backbone","resnet50"),
                             pretrained_backbone=False, out_channels=3).to(device)
        m.load_state_dict(ckpt["model_state_dict"]); m.eval()
        print(f"  Loaded DeepLabV3"); return m
    except Exception as e:
        print(f"  [SKIP] DeepLabV3 error: {e}"); return None


def load_deeplabv3plus(device):
    try:
        from architecture_team_5.deeplabv3plus_model import build_deeplabv3plus
        p = os.path.join(ROOT, "architecture_team_5", "deeplabv3plus",
                         "runs_deeplabv3plus", "best_deeplabv3plus.pt")
        if not os.path.exists(p):
            print(f"  [SKIP] DeepLabV3+ not found: {p}"); return None
        ckpt = torch.load(p, map_location=device)
        m = build_deeplabv3plus(backbone=ckpt.get("backbone","resnet50"),
                                 pretrained_backbone=False, out_channels=3,
                                 output_stride=ckpt.get("output_stride",16)).to(device)
        m.load_state_dict(ckpt["model_state_dict"]); m.eval()
        print(f"  Loaded DeepLabV3+"); return m
    except Exception as e:
        print(f"  [SKIP] DeepLabV3+ error: {e}"); return None


def load_cellpose():
    try:
        from cellpose import models as cp
        gpu  = torch.cuda.is_available()
        mdir = os.path.join(ROOT, "backend", "models")
        chrom = None
        for name in ["cp_chromo_aug", "cp_chromo_no_aug"]:
            pp = os.path.join(mdir, name)
            if os.path.exists(pp):
                chrom = cp.CellposeModel(gpu=gpu, pretrained_model=pp)
                print(f"  Loaded Cellpose chromo: {name}"); break
        nuc = None
        np_ = os.path.join(mdir, "cp_nucleus")
        if os.path.exists(np_):
            nuc = cp.CellposeModel(gpu=gpu, pretrained_model=np_)
            print("  Loaded Cellpose nucleus")
        else:
            nuc = cp.CellposeModel(gpu=gpu, model_type="nuclei")
            print("  Using pretrained Cellpose nuclei fallback")
        return chrom, nuc
    except Exception as e:
        print(f"  [SKIP] Cellpose error: {e}"); return None, None


# ════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ════════════════════════════════════════════════════════════════════════════

def infer_unetpp(model, img_t, device):
    return predict_tta(model, img_t.unsqueeze(0).to(device), device)

def infer_smp(model, img_t, device, three_ch=False):
    b = img_t.unsqueeze(0).to(device)
    if three_ch: b = repeat_3ch(b)
    with torch.no_grad():
        return torch.argmax(model(b), 1)[0].cpu().numpy().astype(np.uint8)

def infer_cellpose(chrom, nuc, cid, data_root, target_size):
    """
    Run Cellpose on the ORIGINAL full-resolution TIFF, not the resized tensor.
    Cellpose estimates cell diameter automatically — feeding it a 256x256 image
    where the nucleus fills the whole frame breaks that estimation entirely,
    causing nucleus Dice to collapse to 0.000. Using the original image fixes this.
    The output mask is resized back to target_size afterwards to match GT.
    """
    import tifffile as tiff_io

    img_path = os.path.join(data_root, "Microscopy_images",
                            f"Microscope_image_Cell{cid}.tif")
    raw = tiff_io.imread(img_path).astype(np.float32)

    # percentile normalise then convert to uint8 (same as dataset pipeline)
    lo, hi = np.percentile(raw, 1), np.percentile(raw, 99.5)
    raw = np.clip(raw, lo, hi)
    raw = ((raw - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)

    # collapse to 2D if needed
    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

    orig_h, orig_w = raw.shape
    sem = np.zeros((orig_h, orig_w), dtype=np.uint8)

    if chrom is not None:
        cm, _, _ = chrom.eval(raw, diameter=None, channels=[0, 0])
        sem[np.asarray(cm) > 0] = 2

    # nucleus: trying custom model first and then falling back to pretrained if it gives 0 instances
    nuc_instances = None
    if nuc is not None:
        try:
            nm, _, _ = nuc.eval(raw, diameter=None, channels=[0, 0],
                                flow_threshold=0.4, cellprob_threshold=-2.0)
            nuc_instances = np.asarray(nm)
        except Exception:
            nuc_instances = None

    # fallback to pretrained nuclei model if custom gave 0 instances
    if nuc_instances is None or np.max(nuc_instances) == 0:
        try:
            from cellpose import models as _cp
            _fallback = _cp.CellposeModel(gpu=False, model_type="nuclei")
            fb, _, _ = _fallback.eval(raw, diameter=None, channels=[0, 0],
                                      flow_threshold=0.4, cellprob_threshold=-2.0)
            nuc_instances = np.asarray(fb)
        except Exception:
            nuc_instances = None

    # set nucleoplasm pixels = all nucleus pixels that are not chromocenter
    if nuc_instances is not None and np.max(nuc_instances) > 0:
        nuc_region = (nuc_instances > 0)
        sem[nuc_region & (sem != 2)] = 1

    # resize output mask to target_size to match GT dimensions
    th, tw = target_size
    if (orig_h, orig_w) != (th, tw):
        sem = cv2.resize(sem, (tw, th), interpolation=cv2.INTER_NEAREST)

    return sem


# ════════════════════════════════════════════════════════════════════════════
# FIGURE GENERATORS
# ════════════════════════════════════════════════════════════════════════════

def fig_bar_charts(summary_df):
    """Grouped bar chart for Dice and IoU."""
    for metric, c1, c2, title in [
        ("Dice", "mean_dice_nuc", "mean_dice_chrom", "Mean Dice Score by Model and Class"),
        ("IoU",  "mean_iou_nuc",  "mean_iou_chrom",  "Mean IoU Score by Model and Class"),
    ]:
        models = summary_df["model"].tolist()
        v1 = summary_df[c1].tolist()
        v2 = summary_df[c2].tolist()
        x  = np.arange(len(models))
        w  = 0.35
        fig, ax = plt.subplots(figsize=(max(7, len(models)*1.5), 5))
        b1 = ax.bar(x-w/2, v1, w, label="Nucleoplasm",  color=NUC_COLOR/255,   alpha=0.88, edgecolor="white")
        b2 = ax.bar(x+w/2, v2, w, label="Chromocenter", color=CHROM_COLOR/255, alpha=0.88, edgecolor="white")
        for bar in list(b1)+list(b2):
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.006, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=18, ha="right", fontsize=10)
        ax.set_ylabel(f"Mean {metric}", fontsize=11)
        ax.set_ylim(0, 1.12)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.legend(fontsize=10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
        plt.tight_layout()
        path = os.path.join(FIG_DIR, f"bar_{metric.lower()}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved: {os.path.basename(path)}")


def fig_boxplots(per_image_df, available_models):
    """Box plot showing score distribution per model."""
    for col, label in [("dice_chrom", "Chromocenter"), ("dice_nuc", "Nucleoplasm")]:
        data   = [per_image_df[per_image_df["model"]==m][col].dropna().tolist() for m in available_models]
        colors = [MODEL_COLORS.get(m,"#888") for m in available_models]
        fig, ax = plt.subplots(figsize=(max(7, len(available_models)*1.5), 5))
        bp = ax.boxplot(data, patch_artist=True, notch=False, widths=0.55,
                        medianprops=dict(color="white", linewidth=2.5))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.82)
        ax.set_xticklabels(available_models, rotation=18, ha="right", fontsize=10)
        ax.set_ylabel("Dice Score", fontsize=11)
        ax.set_title(f"{label} Dice Distribution Across Validation Images",
                     fontsize=12, fontweight="bold", pad=10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True); ax.set_ylim(-0.05, 1.1)
        plt.tight_layout()
        path = os.path.join(FIG_DIR, f"boxplot_dice_{label.lower()}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved: {os.path.basename(path)}")


def fig_scatter(per_image_df, available_models):
    """Per-image chrom Dice: reference model vs every other model."""
    if len(available_models) < 2:
        return
    ref    = "U-Net++" if "U-Net++" in available_models else available_models[0]
    others = [m for m in available_models if m != ref]
    if not others:
        return
    ncols = min(len(others), 4)
    nrows = (len(others) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5*ncols, 4.2*nrows), squeeze=False)
    ref_s = per_image_df[per_image_df["model"]==ref].set_index("cell_id")["dice_chrom"]
    for idx, m in enumerate(others):
        ax  = axes[idx//ncols][idx%ncols]
        sub = per_image_df[per_image_df["model"]==m].set_index("cell_id")["dice_chrom"]
        common = ref_s.index.intersection(sub.index)
        x, y   = ref_s[common].values, sub[common].values
        ax.scatter(x, y, c=MODEL_COLORS.get(m,"#888"), alpha=0.72, s=50,
                   edgecolors="white", linewidths=0.5)
        lo = min(x.min(), y.min()) - 0.05
        hi = max(x.max(), y.max()) + 0.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
        ax.set_xlabel(f"{ref} Chrom Dice", fontsize=9)
        ax.set_ylabel(f"{m} Chrom Dice", fontsize=9)
        ax.set_title(f"{ref} vs {m}", fontsize=10, fontweight="bold")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    for idx in range(len(others), nrows*ncols):
        axes[idx//ncols][idx%ncols].set_visible(False)
    fig.suptitle("Per-Image Chromocenter Dice: Pairwise Comparison",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "scatter_chrom_dice.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")



def fig_confusion_matrices(confusion_data):
    """Normalised pixel confusion matrix heatmap per model."""
    for name, cm in confusion_data.items():
        row_sums  = cm.sum(axis=1, keepdims=True).astype(float)
        cm_norm   = np.where(row_sums > 0, cm/(row_sums+1e-12), 0.0)
        fig, ax   = plt.subplots(figsize=(5.5, 4.5))
        im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        for r in range(3):
            for c in range(3):
                val = cm_norm[r, c]
                ax.text(c, r, f"{val:.2f}\n({cm[r,c]:,})",
                        ha="center", va="center", fontsize=9,
                        color="white" if val > 0.5 else "black")
        ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
        ax.set_xticklabels(CLASS_NAMES, fontsize=9)
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax.set_xlabel("Predicted Class", fontsize=10)
        ax.set_ylabel("True Class", fontsize=10)
        ax.set_title(f"Pixel Confusion Matrix — {name}", fontsize=11, fontweight="bold", pad=8)
        plt.tight_layout()
        safe = name.replace("+","plus").replace("/","_").replace(" ","_")
        path = os.path.join(FIG_DIR, f"confusion_{safe}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved: {os.path.basename(path)}")


def save_comparison_figure(cell_id, orig_u8, gt, pred_dict, score_dict, out_path):
    """
    Two-row grid per cell:
      Row 1: Original | GT | pred(model1) | pred(model2) | ...
      Row 2: (empty)  | (empty) | error(model1) | error(model2) | ...
    """
    model_names = list(pred_dict.keys())
    n_cols = 2 + len(model_names)
    fig = plt.figure(figsize=(3.0*n_cols, 6.8))
    gs  = gridspec.GridSpec(2, n_cols, figure=fig,
                            hspace=0.38, wspace=0.06, height_ratios=[1, 1])
    fig.suptitle(f"Cell {cell_id}", fontsize=11, fontweight="bold")

    # Original
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(orig_u8, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Original", fontsize=8, fontweight="bold"); ax.axis("off")

    # Ground truth
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(colorize_mask(gt))
    ax.set_title("Ground Truth", fontsize=8, fontweight="bold"); ax.axis("off")

    # Bottom-left blank cells
    for bc in [0, 1]:
        fig.add_subplot(gs[1, bc]).axis("off")

    for ci, mname in enumerate(model_names, start=2):
        pred = pred_dict[mname]
        sc   = score_dict.get(mname, {})
        d1   = sc.get("dice_nuc",   float("nan"))
        d2   = sc.get("dice_chrom", float("nan"))

        ax_top = fig.add_subplot(gs[0, ci])
        ax_bot = fig.add_subplot(gs[1, ci])

        if pred is None:
            for ax in (ax_top, ax_bot):
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11, color="grey")
                ax.set_facecolor("#eee"); ax.axis("off")
            ax_top.set_title(mname, fontsize=8); continue

        ax_top.imshow(colorize_mask(pred))
        ax_top.set_title(f"{mname}\nNuc:{d1:.3f}  Chr:{d2:.3f}",
                         fontsize=7.5, fontweight="bold",
                         color=MODEL_COLORS.get(mname, "black"))
        ax_top.axis("off")

        ax_bot.imshow(error_map(pred, gt))
        ax_bot.set_title("Errors (red)", fontsize=7.5)
        ax_bot.axis("off")

    patches = [
        mpatches.Patch(color=np.array(BG_COLOR)/255,    label="Background"),
        mpatches.Patch(color=np.array(NUC_COLOR)/255,   label="Nucleoplasm"),
        mpatches.Patch(color=np.array(CHROM_COLOR)/255, label="Chromocenter"),
        mpatches.Patch(color=np.array(ERR_COLOR)/255,   label="Error pixel"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               fontsize=8, bbox_to_anchor=(0.5, -0.04))
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ════════════════════════════════════════════════════════════════════════════

def compute_stats(per_image_df, available_models):
    rows = []
    for i, m1 in enumerate(available_models):
        for m2 in available_models[i+1:]:
            for label, col in [("Dice Chrom","dice_chrom"), ("Dice Nuc","dice_nuc"),
                                ("IoU Chrom", "iou_chrom"),  ("IoU Nuc", "iou_nuc")]:
                s1 = per_image_df[per_image_df["model"]==m1].set_index("cell_id")[col]
                s2 = per_image_df[per_image_df["model"]==m2].set_index("cell_id")[col]
                common = s1.index.intersection(s2.index)
                if len(common) < 5:
                    continue
                a, b = s1[common].values, s2[common].values
                try:    stat, pval = stats.wilcoxon(a, b)
                except: stat, pval = float("nan"), float("nan")
                rows.append({"model_A": m1, "model_B": m2, "metric": label,
                             "mean_A": round(float(np.mean(a)),4),
                             "mean_B": round(float(np.mean(b)),4),
                             "diff(A-B)": round(float(np.mean(a)-np.mean(b)),4),
                             "wilcoxon_stat": round(float(stat),4),
                             "p_value": round(float(pval),5),
                             "significant_p05": "yes" if pval < 0.05 else "no"})
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    cfg    = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*65}")
    print("Quorum — Full Report Comparison Suite")
    print(f"{'='*65}")
    print(f"Device : {device}")
    print(f"Data   : {cfg['data_root']}")
    print(f"Out    : {OUT_DIR}\n")

    # Dataset
    val_ds = build_val_dataset(cfg)
    print(f"Validation images: {len(val_ds)}\n")

    # Loading models
    print("Loading models...")
    unetpp        = load_unetpp(device)
    unet          = load_unet(device)
    deeplabv3     = load_deeplabv3(device)
    deeplabv3plus = load_deeplabv3plus(device)
    cp_chrom, cp_nuc = load_cellpose()

    MODEL_DEFS = {
        "U-Net++"   : (lambda t: infer_unetpp(unetpp, t, device),                     unetpp is not None),
        "U-Net"     : (lambda t: infer_smp(unet, t, device),                          unet is not None),
        "Cellpose"  : (lambda t, _cid=None: infer_cellpose(cp_chrom, cp_nuc, _cid, cfg["data_root"], tuple(cfg.get("target_size",[256,256]))), (cp_chrom or cp_nuc) is not None),
        "DeepLabV3" : (lambda t: infer_smp(deeplabv3, t, device, three_ch=True),      deeplabv3 is not None),
        "DeepLabV3+": (lambda t: infer_smp(deeplabv3plus, t, device, three_ch=True),  deeplabv3plus is not None),
    }
    available_models = [k for k, v in MODEL_DEFS.items() if v[1]]
    print(f"\nAvailable: {available_models}")
    if not available_models:
        print("No checkpoints found. Exiting."); return

    # Inference loop
    print(f"\nRunning inference on {len(val_ds)} images...")
    per_image_rows = []
    all_preds      = {}
    all_imgs       = {}
    confusion_data = {m: np.zeros((3,3), dtype=np.int64) for m in available_models}

    for i in range(len(val_ds)):
        img_t, gt_t = val_ds[i]
        cid = val_ds.samples[i]
        gt  = gt_t.numpy().astype(np.uint8)
        all_imgs[cid]  = (to_uint8(img_t), gt)
        all_preds[cid] = {}

        for mname in available_models:
            try:
                pred = MODEL_DEFS[mname][0](img_t, cid) if mname == "Cellpose" else MODEL_DEFS[mname][0](img_t)
                if pred.shape != gt.shape:
                    pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
            except Exception as e:
                print(f"  [ERROR] {mname} cell {cid}: {e}"); pred = None

            all_preds[cid][mname] = pred
            if pred is not None:
                d1,i1 = dice_iou(pred, gt, 1)
                d2,i2 = dice_iou(pred, gt, 2)
                confusion_data[mname] += confusion_matrix_pixels(pred, gt)
            else:
                d1=i1=d2=i2=float("nan")
            per_image_rows.append(dict(cell_id=cid, model=mname,
                                       dice_nuc=d1, iou_nuc=i1,
                                       dice_chrom=d2, iou_chrom=i2))

        if (i+1) % 10 == 0 or (i+1) == len(val_ds):
            print(f"  {i+1}/{len(val_ds)} done")

    # CSVs
    per_image_df = pd.DataFrame(per_image_rows)
    per_image_df.to_csv(os.path.join(OUT_DIR, "per_image_metrics.csv"), index=False)
    print("\nSaved: per_image_metrics.csv")

    summary_rows = []
    for m in available_models:
        sub = per_image_df[per_image_df["model"]==m]
        summary_rows.append({
            "model"          : m,
            "mean_dice_nuc"  : round(sub["dice_nuc"].mean(),   4),
            "mean_iou_nuc"   : round(sub["iou_nuc"].mean(),    4),
            "mean_dice_chrom": round(sub["dice_chrom"].mean(), 4),
            "mean_iou_chrom" : round(sub["iou_chrom"].mean(),  4),
            "std_dice_chrom" : round(sub["dice_chrom"].std(),  4),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"), index=False)
    print("Saved: metrics_summary.csv")
    print("\n── Summary ──")
    print(summary_df.to_string(index=False, float_format="%.4f"))

    stats_df = compute_stats(per_image_df, available_models)
    if not stats_df.empty:
        stats_df.to_csv(os.path.join(OUT_DIR, "statistical_tests.csv"), index=False)
        print(f"\nSaved: statistical_tests.csv  ({len(stats_df)} tests)")

    cm_rows = []
    for m, cm in confusion_data.items():
        for r in range(3):
            for c in range(3):
                cm_rows.append({"model":m, "true_class":CLASS_NAMES[r],
                                 "pred_class":CLASS_NAMES[c], "pixel_count":int(cm[r,c])})
    pd.DataFrame(cm_rows).to_csv(os.path.join(OUT_DIR, "confusion_matrices.csv"), index=False)
    print("Saved: confusion_matrices.csv")

    # Figures
    print("\nGenerating figures...")
    fig_bar_charts(summary_df)
    fig_boxplots(per_image_df, available_models)
    fig_scatter(per_image_df, available_models)
    fig_confusion_matrices(confusion_data)


    # Comparison grids — 5 representative cells
    print("  Generating comparison grids...")
    ref    = "U-Net++" if "U-Net++" in available_models else available_models[0]
    ranked = (per_image_df[per_image_df["model"]==ref]
              .set_index("cell_id")["dice_chrom"].dropna().sort_values())
    n   = len(ranked)
    sel = list(dict.fromkeys([ranked.index[q] for q in
                               [0, n//4, n//2, 3*n//4, n-1] if 0 <= q < n]))

    for cid in sel:
        img_u8, gt = all_imgs[cid]
        pred_dict  = all_preds[cid]
        score_dict = {}
        for mname in available_models:
            sub = per_image_df[(per_image_df["model"]==mname) &
                               (per_image_df["cell_id"]==cid)]
            if len(sub):
                r = sub.iloc[0]
                score_dict[mname] = {"dice_nuc": r["dice_nuc"],
                                     "dice_chrom": r["dice_chrom"]}
        out = os.path.join(FIG_DIR, f"comparison_cell{cid}.png")
        save_comparison_figure(cid, img_u8, gt, pred_dict, score_dict, out)
        print(f"  Saved: comparison_cell{cid}.png  "
              f"(chrom Dice {ref}={ranked.get(cid,float('nan')):.3f})")

    # Summary 
    print(f"\n{'='*65}")
    print(f"Done in {int(time.time()-t0)}s.")
    print(f"\nAll outputs: {OUT_DIR}")
    print(f"\n  CSVs for report:")
    print(f"    metrics_summary.csv     ← Table 1  (Dice + IoU per model)")
    print(f"    statistical_tests.csv   ← Wilcoxon p-values (A1 vs A2 claim)")
    print(f"    per_image_metrics.csv   ← full data / appendix")
    print(f"    confusion_matrices.csv  ← pixel-level confusion data")
    print(f"\n  Figures for report:")
    print(f"    bar_dice.png / bar_iou.png          ← main quantitative result")
    print(f"    boxplot_dice_chromocenter.png        ← variance / robustness")
    print(f"    boxplot_dice_nucleoplasm.png")
    print(f"    scatter_chrom_dice.png               ← per-image pairwise")
    print(f"    confusion_<model>.png                ← where each model fails")
    print(f"    comparison_cell*.png                 ← Original|GT|preds|errors")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()