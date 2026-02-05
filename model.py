import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, AutoConfig, EncoderDecoderModel
from dataclasses import dataclass
from typing import Optional, Dict, List

from sequence_model import TextCNN
from gragh_model import GINConvNet
from multimodal_fusion import IntraFusion, TriModalFusion

# ===== Ablation config =====
from dataclasses import dataclass

@dataclass
class AblationConfig:
    use_cmp1d: bool = True   
    use_prot1d: bool = True        
    use_cmp_graph: bool = True     
    use_prot_graph: bool = True   
    use_cmp_3d: bool = True        
    use_prot_3d: bool = True       
    use_intrafusion_seq: bool = True
    use_intrafusion_graph: bool = True
    use_intrafusion_3d: bool = True
    use_cross_fuse: bool = True
    learn_modality_weights: bool = True
    head_use_sim3: bool = True
    head_use_concat: bool = True
    head_use_absdiffs: bool = True
    use_contrastive: bool = True

# ========== 数据集定义 ==========
class SmilesDataset(Dataset):

    def __init__(self, csv_file: str):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        return {
            'cmp_smiles': row['SMILES'],
            'p1_id': row['protein1'],
            'p2_id': row['protein2'],
            'label': torch.tensor(row['label'], dtype=torch.float)
        }
    

class TriComplexClassifier(nn.Module):
    MOLFORMER = "ibm-research/MoLFormer-XL-both-10pct"

    def __init__(self, d_seq: int = 128, d_graph: int = 128, d_3d: int = 128,
                 heads: int = 8, n_layers: int = 2, dropout: float = 0.2,
                 use_checkpoint: str = 'auto', ckpt_threshold_mb: float = 200.0,
                 hadamard_reduce_dim: Optional[int] = None,
                 enable_smiles_cache: bool = False, smiles_cache: Optional[Dict[str, torch.Tensor]] = None,
                 prot1d_dim: int = 21,
                 ablate: AblationConfig = AblationConfig()):
        super().__init__()
        self.ablate = ablate
        self.use_checkpoint = use_checkpoint
        self.ckpt_threshold_mb = ckpt_threshold_mb
        self._ckpt_stats = {f'CP{i}': False for i in range(1, 7)}
        self.hadamard_reduce_dim = hadamard_reduce_dim

        self.enable_smiles_cache = enable_smiles_cache
        self.smiles_cache = smiles_cache if smiles_cache is not None else ({} if enable_smiles_cache else None)

        # ---- MoLFormer----
        self.tokenizer = AutoTokenizer.from_pretrained(self.MOLFORMER, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(self.MOLFORMER, trust_remote_code=True)
        is_encdec = any("EncoderDecoderModel" in a for a in (cfg.architectures or []))
        if is_encdec:
            encdec = EncoderDecoderModel.from_pretrained(self.MOLFORMER, trust_remote_code=True)
            self.molf = encdec.encoder
            hidden_size = encdec.config.encoder.hidden_size
            del encdec
        else:
            self.molf = AutoModel.from_pretrained(self.MOLFORMER, trust_remote_code=True)
            hidden_size = self.molf.config.hidden_size
        for p in self.molf.parameters():
            p.requires_grad = False
        self.seq_proj = nn.Linear(hidden_size, d_seq)

        # ---- TextCNN ----
        self.prot1d_dim = int(prot1d_dim)
        self.prot_textcnn = TextCNN(
            in_dim=self.prot1d_dim, proj_out=d_seq,
            num_filters=64, kernel_sizes=(3, 5, 7), dropout=dropout
        ).to(device)

        # ---- 2D  ----
        self.cmp_gnn = GINConvNet(num_feat=44, out_dim=d_graph).to(device)
        self.prot_gnn = GINConvNet(num_feat=41, out_dim=d_graph).to(device)

        # ---- 3D  ----
        self.cmp_3d_lin  = nn.Linear(768,  d_3d).to(device) 
        self.prot_3d_lin = nn.Linear(1280, d_3d).to(device)  

        # ---- IntraFusion / CrossFusion ----
        self.fuse_seq   = IntraFusion(d_seq,   n_layers, heads, pf_ratio=4, dp=dropout).to(device)
        self.fuse_graph = IntraFusion(d_graph, n_layers, heads, pf_ratio=4, dp=dropout).to(device)
        self.fuse_3d    = IntraFusion(d_3d,    n_layers, heads, pf_ratio=4, dp=dropout).to(device)
        self.cross_fuse = TriModalFusion(d_emb=d_seq, pf_ratio=4, dropout=dropout, n_layers=n_layers).to(device)

        # ---- Hybrid ----
        hybrid_dim = 0
        if self.ablate.head_use_sim3:    hybrid_dim += 3
        if self.ablate.head_use_concat:  hybrid_dim += 3 * d_seq
        if self.ablate.head_use_absdiffs:hybrid_dim += 3 * d_seq
        if hybrid_dim == 0:
            hybrid_dim = 3 * d_seq

        if hadamard_reduce_dim is not None:
            self.hadamard_proj = nn.Sequential(
                nn.LayerNorm(hybrid_dim),
                nn.Linear(hybrid_dim, hadamard_reduce_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ).to(device)
            clf_in = hadamard_reduce_dim
        else:
            self.hadamard_proj = None
            clf_in = hybrid_dim

        self.classifier = nn.Sequential(
            nn.Linear(clf_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        ).to(device)

        if self.ablate.learn_modality_weights:
            self.modality_weights = nn.Parameter(torch.ones(3, device=device))
        else:
            self.register_buffer("fixed_weights", torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32))

        self.to(device)

    # ===== utilities =====
    @staticmethod
    def _approx_tensor_mb(*args):
        total = 0.0
        for a in args:
            if isinstance(a, torch.Tensor):
                total += a.numel() * (a.element_size() if a.dtype != torch.bool else 1)
        return total / (1024 ** 2)

    def _should_ckpt(self, *args) -> bool:
        if not self.training: return False
        if self.use_checkpoint == 'never': return False
        if self.use_checkpoint == 'always': return True
        has_grad = any(isinstance(a, torch.Tensor) and a.requires_grad for a in args)
        if not has_grad: return False
        return self._approx_tensor_mb(*args) >= float(self.ckpt_threshold_mb)

    def _ckpt(self, tag: str, fn, *args):
        if self._should_ckpt(*args):
            try:
                out = torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
            except TypeError:
                out = torch.utils.checkpoint.checkpoint(fn, *args)
            self._ckpt_stats[tag] = True
            return out
        return fn(*args)

    def ckpt_stats(self): return {k: bool(v) for k, v in self._ckpt_stats.items()}

    def _prot1d_preproc(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return F.one_hot(x.long(), num_classes=self.prot1d_dim).float()
        elif x.dim() == 3:
            assert x.shape[-1] == self.prot1d_dim, f"expected last dim {self.prot1d_dim}, got {x.shape[-1]}"
            return x.float()
        else:
            raise RuntimeError(f"protein 1D must be [B,L] or [B,L,{self.prot1d_dim}], got {tuple(x.shape)}")

    @staticmethod
    def _normalize(z): return F.normalize(z, dim=1)

    def info_nce_loss(self, z1, z2, tau=0.1):
        z1, z2 = self._normalize(z1), self._normalize(z2)
        pos = torch.exp((z1 * z2).sum(dim=1) / tau)
        all_sim = torch.exp(z1 @ z2.T / tau)
        neg = all_sim.sum(dim=1) - pos
        return (-torch.log(pos / neg)).mean()

    def _ensure_graph_tensor(self, raw):
        if torch.is_tensor(raw):
            return raw if raw.dim() == 2 else raw.mean(dim=1)
        if isinstance(raw, (tuple, list)):
            for t in raw:
                if torch.is_tensor(t) and t.dim() == 2: return t
            for t in raw:
                if torch.is_tensor(t) and t.dim() == 3: return t.mean(dim=1)
        if isinstance(raw, dict):
            for k in ("graph","graph_emb","pool","readout"):
                v = raw.get(k, None)
                if torch.is_tensor(v) and v.dim()==2: return v
            for k in ("node","nodes","node_repr","nodes_repr"):
                v = raw.get(k, None)
                if torch.is_tensor(v) and v.dim()==3: return v.mean(dim=1)
            for v in raw.values():
                if torch.is_tensor(v):
                    return v if v.dim()==2 else (v.mean(dim=1) if v.dim()==3 else None)
        raise TypeError(f"Unsupported GNN output: {type(raw)}")

    def _weighted_fuse(self, reps: List[torch.Tensor]):
        if self.ablate.learn_modality_weights:
            w = torch.softmax(self.modality_weights, dim=0)
        else:
            w = self.fixed_weights.to(reps[0].device).to(reps[0].dtype)
        return w[0]*reps[0] + w[1]*reps[1] + w[2]*reps[2]

    # ===== forward =====
    def forward(self,
                cmp_smiles, cmp_g, cmp_3d,
                p1_w2v, p1_g, p1_3d,
                p2_w2v, p2_g, p2_3d,
                return_contrastive=False):

        B = p1_w2v.size(0)

        # ---- compound 1D ----
        if self.ablate.use_cmp1d:
            cls_emb = self._encode_smiles_cls(cmp_smiles)
            def _cmp_1d_block(x): return self.seq_proj(x)
            cmp1d_in = cls_emb.detach()
            if self.training: cmp1d_in = cmp1d_in.requires_grad_(True)
            c_seq = self._ckpt('CP1', _cmp_1d_block, cmp1d_in)   # [B, d_seq]
        else:
            c_seq = torch.zeros(B, self.seq_proj.out_features, device=device)

        # ---- compound 2D  ----
        if self.ablate.use_cmp_graph:
            def _cmp_2d_block(anchor):
                raw = self.cmp_gnn(cmp_g)
                gemb = self._ensure_graph_tensor(raw)
                return gemb + anchor.sum() * 0
            anchor = torch.ones(1, device=device, requires_grad=True) if self.training else torch.ones(1, device=device)
            c_graph = self._ckpt('CP2', _cmp_2d_block, anchor)  # [B, d_graph]
        else:
            c_graph = torch.zeros(B, self.fuse_graph.d_model, device=device)  # d_graph

        # ---- compound 3D ----
        if self.ablate.use_cmp_3d:
            def _cmp_3d_block(x): return self.cmp_3d_lin(x)
            cmp_3d_in = cmp_3d.squeeze(1)
            if self.training: cmp_3d_in = cmp_3d_in.detach().requires_grad_(True)
            c_3d = self._ckpt('CP3', _cmp_3d_block, cmp_3d_in)  # [B, d_3d]
        else:
            c_3d = torch.zeros(B, self.fuse_3d.d_model, device=device)       # d_3d

        # ---- protein 1D ----
        if self.ablate.use_prot1d:
            def _prot_1d_block(x):
                x_oh = self._prot1d_preproc(x)
                return self.prot_textcnn(x_oh)
            p1_1d_in = p1_w2v; p2_1d_in = p2_w2v
            if self.training:
                p1_1d_in = p1_1d_in.detach().requires_grad_(True)
                p2_1d_in = p2_1d_in.detach().requires_grad_(True)
            p1_seq = self._ckpt('CP4', _prot_1d_block, p1_1d_in)
            p2_seq = self._ckpt('CP4', _prot_1d_block, p2_1d_in)
        else:
            p1_seq = torch.zeros(B, self.fuse_seq.d_model, device=device)
            p2_seq = torch.zeros(B, self.fuse_seq.d_model, device=device)

        # ---- protein 2D  ----
        if self.ablate.use_prot_graph:
            def _prot_2d_block(anchor, g):
                raw = self.prot_gnn(g)
                ge = self._ensure_graph_tensor(raw)
                return ge + anchor.sum() * 0
            p_anchor = torch.ones(1, device=device, requires_grad=True) if self.training else torch.ones(1, device=device)
            p1_graph = self._ckpt('CP5', lambda a: _prot_2d_block(a, p1_g), p_anchor)
            p2_graph = self._ckpt('CP5', lambda a: _prot_2d_block(a, p2_g), p_anchor)
        else:
            p1_graph = torch.zeros(B, self.fuse_graph.d_model, device=device)
            p2_graph = torch.zeros(B, self.fuse_graph.d_model, device=device)

        # ---- protein 3D ----
        if self.ablate.use_prot_3d:
            def _prot_3d_block(x): return self.prot_3d_lin(x)
            p1_3d_in = p1_3d.mean(dim=1); p2_3d_in = p2_3d.mean(dim=1)
            if self.training:
                p1_3d_in = p1_3d_in.detach().requires_grad_(True)
                p2_3d_in = p2_3d_in.detach().requires_grad_(True)
            p1_3d_proj = self._ckpt('CP6', _prot_3d_block, p1_3d_in)
            p2_3d_proj = self._ckpt('CP6', _prot_3d_block, p2_3d_in)
        else:
            p1_3d_proj = torch.zeros(B, self.fuse_3d.d_model, device=device)
            p2_3d_proj = torch.zeros(B, self.fuse_3d.d_model, device=device)

        # ---- IntraFusion + weighted ----
        c_seq_f   = self.fuse_seq  (c_seq.unsqueeze(1))   if self.ablate.use_intrafusion_seq   else c_seq
        c_graph_f = self.fuse_graph(c_graph.unsqueeze(1)) if self.ablate.use_intrafusion_graph else c_graph
        c_3d_f    = self.fuse_3d   (c_3d.unsqueeze(1))    if self.ablate.use_intrafusion_3d    else c_3d
        v_c  = self._weighted_fuse([c_seq_f, c_graph_f, c_3d_f])

        p1_seq_f   = self.fuse_seq  (p1_seq.unsqueeze(1))   if self.ablate.use_intrafusion_seq   else p1_seq
        p1_graph_f = self.fuse_graph(p1_graph.unsqueeze(1)) if self.ablate.use_intrafusion_graph else p1_graph
        p1_3d_f    = self.fuse_3d   (p1_3d_proj.unsqueeze(1)) if self.ablate.use_intrafusion_3d else p1_3d_proj
        v_p1 = self._weighted_fuse([p1_seq_f, p1_graph_f, p1_3d_f])

        p2_seq_f   = self.fuse_seq  (p2_seq.unsqueeze(1))   if self.ablate.use_intrafusion_seq   else p2_seq
        p2_graph_f = self.fuse_graph(p2_graph.unsqueeze(1)) if self.ablate.use_intrafusion_graph else p2_graph
        p2_3d_f    = self.fuse_3d   (p2_3d_proj.unsqueeze(1)) if self.ablate.use_intrafusion_3d else p2_3d_proj
        v_p2 = self._weighted_fuse([p2_seq_f, p2_graph_f, p2_3d_f])

        # ---- Cross Fusion ----
        if self.ablate.use_cross_fuse:
            def _cross_block(vx):
                out = self.cross_fuse(vx, vx, vx)
                return out[0] if isinstance(out, (tuple, list)) else out
            v_c  = _cross_block(v_c)
            v_p1 = _cross_block(v_p1)
            v_p2 = _cross_block(v_p2)

        # ---- LayerNorm + LN ----
        def _safe_embed(x: torch.Tensor) -> torch.Tensor:
            x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            return nn.LayerNorm(x.shape[-1]).to(x.device)(x)
        v_c  = _safe_embed(v_c)
        v_p1 = _safe_embed(v_p1)
        v_p2 = _safe_embed(v_p2)

        # ---- Hybrid ----
        parts = []
        if self.ablate.head_use_sim3:
            s_cp1  = (v_c  * v_p1).sum(dim=-1, keepdim=True)
            s_cp2  = (v_c  * v_p2).sum(dim=-1, keepdim=True)
            s_p1p2 = (v_p1 * v_p2).sum(dim=-1, keepdim=True)
            parts.append(torch.cat([s_cp1, s_cp2, s_p1p2], dim=-1))  # [B,3]
        if self.ablate.head_use_concat:
            parts.append(torch.cat([v_c, v_p1, v_p2], dim=-1))       # [B,3d]
        if self.ablate.head_use_absdiffs:
            parts.append(torch.cat([(v_c - v_p1).abs(),
                                    (v_c - v_p2).abs(),
                                    (v_p1 - v_p2).abs()], dim=-1))  # [B,3d]
        if not parts:
            parts.append(torch.cat([v_c, v_p1, v_p2], dim=-1))       # 兜底
        hybrid_feat_vec = torch.cat(parts, dim=-1).float()

        feat = self.hadamard_proj(hybrid_feat_vec) if self.hadamard_proj is not None else hybrid_feat_vec
        logits = self.classifier(feat).squeeze(-1)

        if not return_contrastive:
            return logits

        if self.ablate.use_contrastive:
            closs = (
                self.info_nce_loss(v_c, v_p1) +
                self.info_nce_loss(v_c, v_p2) +
                self.info_nce_loss(v_p1, v_p2)
            ) / 3.0
        else:
            closs = torch.tensor(0.0, device=logits.device)

        return logits, closs
