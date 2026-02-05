import os, warnings, torch, numpy as np, pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from sklearn import metrics as _m
import torch

from Dataset import TripleDataset
from model   import TriComplexClassifier

torch.backends.cudnn.enabled = False
warnings.filterwarnings("ignore")
DEVICE = torch.device("cpu")

BATCH      = 16
EPOCHS     = 50
LR         = 1e-4
PATIENCE   = 20
ALPHA      = 0.1        


FOLD_NPZ   = Path("/home/xwl/MG/fold_indices.npz")   
SAVE_DIR   = Path("/home/xwl/MG/results"); SAVE_DIR.mkdir(exist_ok=True, parents=True)  
RESULT_CSV = SAVE_DIR / "cv_results.csv"

PRED_DIR = SAVE_DIR / "val_preds"; PRED_DIR.mkdir(exist_ok=True, parents=True)

# ──────────────────────────────
FULL_CSV = "/home/xwl/MG/dataset.csv"
ds_full  = TripleDataset(FULL_CSV)
NUM_SAMPLES = len(ds_full)

if not FOLD_NPZ.exists():
    raise FileNotFoundError(f"未找到固定划分文件: {FOLD_NPZ}")

npz = np.load(FOLD_NPZ, allow_pickle=True)

idxes = sorted(set(
    int(k.split("_")[1]) for k in npz.files
    if (k.startswith("train_") or k.startswith("val_")) and k.split("_")[1].isdigit()
))

fold_ids = [i for i in idxes if f"train_{i}" in npz.files and f"val_{i}" in npz.files]
if not fold_ids:
    raise ValueError(f"{FOLD_NPZ} 中未发现成对的 train_i/val_i 键。")

splits = []
all_val = set()
for i in sorted(fold_ids):
    tr_idx = np.array(npz[f"train_{i}"], dtype=int)
    val_idx = np.array(npz[f"val_{i}"], dtype=int)

    splits.append((tr_idx, val_idx))
    all_val |= set(val_idx.tolist())

print(f"[Info] 读取固定划分，共 {len(splits)} 折；折号: {sorted(fold_ids)}")


def compute_10_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5):


    if np.isnan(y_prob).any():
        y_prob = np.nan_to_num(y_prob, nan=0.5)

    y_pred = (y_prob >= thr).astype(int)

    acc   = _m.accuracy_score(y_true, y_pred)
    prec  = _m.precision_score(y_true, y_pred, zero_division=0)
    rec   = _m.recall_score(y_true, y_pred, zero_division=0)
    f1    = _m.f1_score(y_true, y_pred, zero_division=0)
    bacc  = _m.balanced_accuracy_score(y_true, y_pred)
    mcc   = _m.matthews_corrcoef(y_true, y_pred)
    kappa = _m.cohen_kappa_score(y_true, y_pred)

    # Specificity
    cm = _m.confusion_matrix(y_true, y_pred, labels=[0,1])
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        spec = 0.0


    if len(np.unique(y_true)) == 2:
        auc  = _m.roc_auc_score(y_true, y_prob)
        ap   = _m.average_precision_score(y_true, y_prob)
    else:
        auc, ap = float("nan"), float("nan")

    return {
        "acc": acc, "prec": prec, "rec": rec, "spec": spec,
        "f1": f1, "bacc": bacc, "auc": auc, "mcc": mcc, "kappa": kappa, "ap": ap
    }

# ============== 训练与验证 ==============
records = []
all_pred_rows = []

for fold_idx, (tr_idx, val_idx) in enumerate(splits, 1):
    print(f"\n========== Fold {fold_idx}/{len(splits)} ==========")

    tr_loader = DataLoader(
        Subset(ds_full, tr_idx),
        batch_size=BATCH, shuffle=True,              
        collate_fn=TripleDataset.collate_fn,
        drop_last=True
    )
    va_loader = DataLoader(
        Subset(ds_full, val_idx),
        batch_size=BATCH, shuffle=False,
        collate_fn=TripleDataset.collate_fn
    )

    sample = ds_full[0]["cmp_3d"]            
    import numpy as np
    a = np.asarray(sample)
    if a.ndim == 1:
        CMP3D_IN = a.shape[0]
    else:
        CMP3D_IN = a.shape[-1]               

    model     = TriComplexClassifier(prot1d_dim=25,dropout=0.1)
    criterion = BCEWithLogitsLoss()
    optim     = Adam(model.parameters(), lr=LR)

    best_auc, epochs_bad = 0.0, 0

    best_epoch   = None
    best_y_true  = None
    best_y_prob  = None
    val_order    = val_idx.copy()



    for ep in range(1, EPOCHS + 1):
        # ========  Train  ========
        model.train()
        tr_loss = 0.0
        for batch in tr_loader:
            (cmp_smiles, cmp_g, cmp_3d_batch,
             p1_w2v, p1_g, p1_3d,
             p2_w2v, p2_g, p2_3d,
             y) = batch

            logits, closs = model(
                cmp_smiles, cmp_g,  cmp_3d_batch,
                p1_w2v,    p1_g,   p1_3d,
                p2_w2v,    p2_g,   p2_3d,
                return_contrastive=True
            )
            bce  = criterion(logits.squeeze(-1), y.squeeze(-1))
            loss = bce + ALPHA * closs

            optim.zero_grad()
            loss.backward()
            optim.step()
            tr_loss += loss.item()

        print(f"Fold {fold_idx} | Epoch {ep:02d} | Train Loss {tr_loss/len(tr_loader):.4f}")

        # ========  Validation  ========
        model.eval()
        y_true, y_prob = [], []
        with torch.no_grad():
            for batch in va_loader:
                (cmp_smiles, cmp_g, cmp_3d_batch,
                 p1_w2v, p1_g, p1_3d,
                 p2_w2v, p2_g, p2_3d,
                 y) = batch

                logits = model(
                    cmp_smiles, cmp_g,  cmp_3d_batch,
                    p1_w2v,    p1_g,   p1_3d,
                    p2_w2v,    p2_g,   p2_3d,
                    return_contrastive=False
                )
                probs = torch.sigmoid(logits).squeeze(-1)

                y_true.append(y.squeeze(-1).cpu())
                y_prob.append(probs.cpu())

        y_true = torch.cat(y_true).numpy().astype(int)
        y_prob = torch.cat(y_prob).numpy()

        met = compute_10_metrics(y_true, y_prob, thr=0.5)

        print(
            f"Fold {fold_idx} | Epoch {ep:02d} | "
            f"AUC {met['auc']:.4f} | AP {met['ap']:.4f} | "
            f"ACC {met['acc']:.4f} | F1 {met['f1']:.4f} | "
            f"PREC {met['prec']:.4f} | REC {met['rec']:.4f} | SPEC {met['spec']:.4f} | "
            f"BACC {met['bacc']:.4f} | MCC {met['mcc']:.4f} | Kappa {met['kappa']:.4f}"
        )

        row = {"fold": fold_idx, "epoch": ep}
        row.update(met)
        records.append(row)

        if met["auc"] > best_auc:
            best_auc, epochs_bad = met["auc"], 0
            best_epoch  = ep
            best_y_true = y_true.copy()
            best_y_prob = y_prob.copy()
            torch.save(model.state_dict(), SAVE_DIR / f"fold{fold_idx}_best.pth")
            print(f"   ↑  New best AUC: {best_auc:.4f} (epoch {best_epoch})")
        else:
            epochs_bad += 1
            if epochs_bad >= PATIENCE:
                print(f"   → Early stopping (no improvement in {PATIENCE} epochs)")
                break

    print(f"========== Fold {fold_idx} done | Best AUC {best_auc:.4f} ==========")

    if best_y_prob is not None:
        df_fold = pd.DataFrame({
            "fold":        fold_idx,
            "best_epoch":  best_epoch,
            "global_idx":  val_order,      
            "y_true":      best_y_true,
            "y_prob":      best_y_prob
        })
        out_csv = PRED_DIR / f"fold{fold_idx}_best_val_pred.csv"
        df_fold.to_csv(out_csv, index=False)
        np.savez(PRED_DIR / f"fold{fold_idx}_best_val_pred.npz",
                 fold=fold_idx, best_epoch=best_epoch,
                 idx=val_order, y_true=best_y_true, y_prob=best_y_prob)
        print(f"Saved best-val predictions to: {out_csv}")
        all_pred_rows.append(df_fold)

pd.DataFrame(records).to_csv(RESULT_CSV, index=False)
print(f"\nAll folds finished. CV results saved to {RESULT_CSV}")
