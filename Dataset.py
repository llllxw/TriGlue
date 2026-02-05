
import warnings, torch, dgl, numpy as np, pandas as pd
from pathlib import Path
from typing import Dict, List
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from dataclasses import dataclass

@dataclass
class DataAugCfg:
    smiles_mask_rate: float = 0.15
    edge_drop: float        = 0.10
    gauss_sigma: float      = 0.02

AUG_ON  = DataAugCfg(0.15, 0.10, 0.02)
AUG_OFF = DataAugCfg(0.0,  0.0,  0.0)
AUG_NO_SMILES = DataAugCfg(0.0,  0.10, 0.02)
AUG_NO_EDGES  = DataAugCfg(0.15, 0.0,  0.02)
AUG_NO_3DNOISE= DataAugCfg(0.15, 0.10, 0.0)

# ───────────────── 路径 ─────────────────
PRE      = Path("/home/xwl/MG/predata")
CMP_GPH  = PRE / "compound_graph"
CMP_3D   = PRE / "compound_3d_embedding"  

PROT_W2V = PRE / "protein_embedding/one_hot"
PROT_GPH = PRE / "protein_graph"
PROT_3D  = PRE / "protein_3d_embedding"   

# ───────────────── MoLFormer tokenizer ─────────────────
MOLFORMER_ID = "ibm-research/MoLFormer-XL-both-10pct"
TOK   = AutoTokenizer.from_pretrained(MOLFORMER_ID, trust_remote_code=True)
MASK_ID = TOK.mask_token_id
VOCAB   = TOK.vocab_size

# ───────────────── util ─────────────────
def _np(p): return np.load(p, allow_pickle=True)
def _g (p): return dgl.load_graphs(str(p))[0][0]

def _pad(arrs):
    proc = []
    for a in arrs:
        a = np.asarray(a)
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        if a.ndim == 1:
            a = a[:, None]
        proc.append(a)

    tail_shapes = {tuple(a.shape[1:]) for a in proc}
    if len(tail_shapes) != 1:
        raise ValueError(
            f"Inconsistent trailing feature dims in batch: {sorted(tail_shapes)}. "
        )

    L_max = max(a.shape[0] for a in proc)
    out = []
    for a in proc:
        pad_width = [(0, L_max - a.shape[0])] + [(0, 0)] * (a.ndim - 1)
        out.append(np.pad(a, pad_width, mode='constant'))
    return np.stack(out, axis=0)

def _aug_emb(x, sigma=0.02): return x + torch.randn_like(x) * sigma

def _aug_g(g, drop=0.1):
    e = g.edges(form='eid')
    keep = torch.rand(len(e)) > drop
    return dgl.edge_subgraph(g, e[keep])

# ───────────── token-mask 增广 ─────────────
def augment_token_ids(ids: torch.Tensor,
                      mask_rate: float = 0.15,
                      vocab_size: int = None) -> torch.Tensor:
    if mask_rate <= 0.0:
        return ids
    if vocab_size is None:
        raise ValueError("augment_token_ids 需要明确的 vocab_size。")

    ids  = ids.clone()
    dev  = ids.device
    prob = torch.rand_like(ids.float())

    mask_pos = prob < mask_rate
    ids[mask_pos & (prob < mask_rate * 0.8)] = MASK_ID
    rand_pos = mask_pos & (prob >= mask_rate * 0.8) & (prob < mask_rate * 0.9)
    if rand_pos.any():
        ids[rand_pos] = torch.randint(
            low=0, high=vocab_size,
            size=(int(rand_pos.sum().item()),),
            device=dev
        )

    return ids

# ══════════════════════════════════════════════════════
class TripleDataset(Dataset):
    def __init__(self,
                 csv_path: str,
                 train: bool = True,
                 smiles_mask_rate: float = 0.15,
                 edge_drop: float = 0.1,
                 gauss_sigma: float = 0.02):
        
        self.df    = pd.read_csv(csv_path)
        self.train = train
        self.smiles_mask_rate = float(smiles_mask_rate)
        self.edge_drop        = float(edge_drop)
        self.gauss_sigma      = float(gauss_sigma)

    def __len__(self): return len(self.df)

    # ------ protein loader ------
    def _load_prot(self, pid):
        return (_np(PROT_W2V/f"{pid}.npy"),
                _g (PROT_GPH/f"{pid}.bin"),
                _np(PROT_3D /f"{pid}.npy"))

    # ------------- single sample -------------
    def __getitem__(self, idx):
        r   = self.df.iloc[idx]
        cid = str(r["ID"])
        p1, p2 = str(r["protein1"]), str(r["protein2"])

        smiles_ids = torch.tensor(
            TOK(r["SMILES"], add_special_tokens=True)["input_ids"],
            dtype=torch.long
        )
        if self.train and self.smiles_mask_rate > 0:
            smiles_ids = augment_token_ids(
                smiles_ids,
                mask_rate=self.smiles_mask_rate,
                vocab_size=VOCAB           
            )
        masked_smiles = TOK.decode(smiles_ids, skip_special_tokens=False).replace(" ", "")

        # --- compound ---
        sample = dict(
            cmp_smiles = masked_smiles,          
            cmp_g      = _g (CMP_GPH/f"{cid}.bin"),
            cmp_3d     = _np(CMP_3D /f"{cid}.npy"),
            y          = float(r["label"]),
            train      = self.train
        )

        # --- protein 1 / 2 ---
        p1_w2v, p1_g, p1_3d = self._load_prot(p1)
        p2_w2v, p2_g, p2_3d = self._load_prot(p2)
        sample.update(dict(
            p1_w2v=p1_w2v, p1_g=p1_g, p1_3d=p1_3d,
            p2_w2v=p2_w2v, p2_g=p2_g, p2_3d=p2_3d
        ))

        if self.train and self.edge_drop > 0:
            sample["cmp_g"] = _aug_g(sample["cmp_g"], drop=self.edge_drop)
            sample["p1_g"]  = _aug_g(sample["p1_g"],  drop=self.edge_drop)
            sample["p2_g"]  = _aug_g(sample["p2_g"],  drop=self.edge_drop)

        return sample

    # ------------- collate -------------
    @staticmethod
    def collate_fn(batch: List[Dict]):
        cmp_smi = [b["cmp_smiles"] for b in batch]

        cmp_3d  = torch.tensor(_pad([b["cmp_3d"] for b in batch]), dtype=torch.float32)
        p1_3d   = torch.tensor(_pad([b["p1_3d"] for b in batch]), dtype=torch.float32)
        p2_3d   = torch.tensor(_pad([b["p2_3d"] for b in batch]), dtype=torch.float32)

        p1_w2v  = torch.tensor(np.stack([b["p1_w2v"] for b in batch]), dtype=torch.float32)
        p2_w2v  = torch.tensor(np.stack([b["p2_w2v"] for b in batch]), dtype=torch.float32)

        def _batch_graph(key):
            gl = [b[key] for b in batch]
            for i, g in enumerate(gl):
                g.ndata["graph_id"] = torch.full((g.num_nodes(),), i, dtype=torch.long)
            return dgl.batch(gl)

        cmp_g = _batch_graph("cmp_g")
        p1_g  = _batch_graph("p1_g")
        p2_g  = _batch_graph("p2_g")

        if batch[0]["train"]:
            for t in (cmp_3d, p1_3d, p2_3d):
                t[:] = _aug_emb(t, sigma=0.02)

        y = torch.tensor([b["y"] for b in batch], dtype=torch.float32).unsqueeze(1)

        return (cmp_smi, cmp_g, cmp_3d,
                p1_w2v, p1_g, p1_3d,
                p2_w2v, p2_g, p2_3d,
                y)



