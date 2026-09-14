import os
if os.environ.get("MG_FORCE_CPU", "0") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 禁用 GPU

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import dgl
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig, EncoderDecoderModel
from typing import Optional, Dict, List, Union

# ✅ Protein 1-D (one-hot + TextCNN)
from sequence_model import TextCNN
from gragh_model import GATNet
from gragh_model import GCNNet
from gragh_model import GINConvNet
# ✅ 2-D graph 改为 graphtransformer（从 gragh_model.py 直接引用）
from gragh_model import GraphTransformer
from multimodal_fusion import IntraFusion, TriModalFusion

device = torch.device("cpu")
# ===== Ablation config =====
from dataclasses import dataclass

@dataclass
class AblationConfig:
    # 分支是否启用
    # use_cmp1d stays off by default for compatibility with existing checkpoints;
    # formal revised full-model runs should set it explicitly.
    use_cmp1d: bool = False        # 化合物 1D (MoLFormer+Linear)
    use_prot1d: bool = True        # 蛋白 1D (one-hot+TextCNN)
    use_cmp_graph: bool = True     # 化合物 GNN
    use_prot_graph: bool = True    # 蛋白 GNN
    use_cmp_3d: bool = True        # Uni-Mol2 compound embedding
    # Legacy field name retained for checkpoint/config compatibility.  This
    # branch is an ESM-2 residue embedding, not a geometric protein-3D input.
    use_prot_3d: bool = True       # ESM-2 pretrained protein sequence embedding

    # IntraFusion（3 路）
    use_intrafusion_seq: bool = True
    use_intrafusion_graph: bool = True
    use_intrafusion_3d: bool = True

    # TriModalFusion（对象内交互）
    use_cross_fuse: bool = True

    # 模态权重：学习 or 固定均匀
    learn_modality_weights: bool = True

    # Hybrid 头组成
    head_use_sim3: bool = True
    head_use_concat: bool = True
    head_use_absdiffs: bool = True
    head_use_hadamards: bool = False

    # 对比学习损失
    use_contrastive: bool = True

    # Corrected fusion protocol.  False preserves legacy checkpoints.  In v2,
    # modalities are tokens within each object, and compound/protein objects
    # are tokens in a second interaction block.
    fusion_v2: bool = False
    use_intra_modal_attention_v2: bool = True
    use_cross_object_attention_v2: bool = True
    use_sample_dynamic_weights_v2: bool = True

# ========== 数据集定义 ==========
class SmilesDataset(Dataset):
    """
    读取 dataset.csv 并返回 SMILES、protein1、protein2、label
    CSV 列: ID,Compound,protein1,protein2,label,SMILES
    """
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
        self.d_seq = int(d_seq)
        self.d_graph = int(d_graph)
        self.d_3d = int(d_3d)
        self.use_checkpoint = use_checkpoint
        self.ckpt_threshold_mb = ckpt_threshold_mb
        self._ckpt_stats = {f'CP{i}': False for i in range(1, 7)}
        self.hadamard_reduce_dim = hadamard_reduce_dim

        self.enable_smiles_cache = enable_smiles_cache
        self.smiles_cache = smiles_cache if smiles_cache is not None else ({} if enable_smiles_cache else None)

        # ---- MoLFormer（化合物 1D）----
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

        # ---- 蛋白 1D：TextCNN ----
        self.prot1d_dim = int(prot1d_dim)
        self.prot_textcnn = TextCNN(
            in_dim=self.prot1d_dim, proj_out=d_seq,
            num_filters=64, kernel_sizes=(3, 5, 7), dropout=dropout
        ).to(device)

        # ---- 2D 图 ----
        self.cmp_gnn = GINConvNet(num_feat=44, out_dim=d_graph).to(device)
        self.prot_gnn = GINConvNet(num_feat=41, out_dim=d_graph).to(device)

        # ---- Continuous embedding projections ----
        self.cmp_3d_lin  = nn.Linear(768,  d_3d).to(device)  # compound: unimol2
        self.prot_3d_lin = nn.Linear(1280, d_3d).to(device)  # ESM-2 (legacy name: prot_3d)
        self._dyn_proj_bank = nn.ModuleDict()

        # ---- IntraFusion / CrossFusion ----
        self.fuse_seq   = IntraFusion(d_seq,   n_layers, heads, pf_ratio=4, dp=dropout).to(device)
        self.fuse_graph = IntraFusion(d_graph, n_layers, heads, pf_ratio=4, dp=dropout).to(device)
        self.fuse_3d    = IntraFusion(d_3d,    n_layers, heads, pf_ratio=4, dp=dropout).to(device)
        self.cross_fuse = TriModalFusion(d_emb=d_seq, pf_ratio=4, dropout=dropout, n_layers=n_layers).to(device)

        # Corrected, opt-in fusion blocks.  They are created only for v2 so
        # legacy parameter initialization and checkpoints remain unchanged.
        if self.ablate.fusion_v2:
            modal_layer = nn.TransformerEncoderLayer(
                d_model=d_seq, nhead=heads, dim_feedforward=4 * d_seq,
                dropout=dropout, activation="gelu", batch_first=True,
                norm_first=True,
            )
            object_layer = nn.TransformerEncoderLayer(
                d_model=d_seq, nhead=heads, dim_feedforward=4 * d_seq,
                dropout=dropout, activation="gelu", batch_first=True,
                norm_first=True,
            )
            self.v2_modal_encoder = nn.TransformerEncoder(modal_layer, num_layers=n_layers).to(device)
            self.v2_object_encoder = nn.TransformerEncoder(object_layer, num_layers=n_layers).to(device)
            self.v2_modal_gate = nn.Sequential(
                nn.Linear(d_seq, d_seq // 2), nn.GELU(), nn.Linear(d_seq // 2, 1)
            ).to(device)

        # ---- Hybrid 分类头 ----
        hybrid_dim = 0
        if self.ablate.head_use_sim3:    hybrid_dim += 3
        if self.ablate.head_use_concat:  hybrid_dim += 3 * d_seq
        if self.ablate.head_use_absdiffs:hybrid_dim += 3 * d_seq
        if self.ablate.head_use_hadamards:hybrid_dim += 3 * d_seq
        if hybrid_dim == 0:
            # 若全部关闭，兜底用 concat
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

        self.projection = nn.Sequential(
            nn.Linear(d_seq, d_seq),
            nn.ReLU(inplace=True),
            nn.Linear(d_seq, d_seq)
        ).to(device)

        if self.ablate.learn_modality_weights:
            self.modality_weights = nn.Parameter(torch.ones(3, device=device))
        else:
            self.register_buffer("fixed_weights", torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32))

        # Registered output normalizers.  The previous implementation created
        # fresh LayerNorm modules inside every forward pass, so their affine
        # parameters were never optimized or saved in checkpoints.
        self.compound_object_norm = nn.LayerNorm(d_seq).to(device)
        self.protein_object_norm = nn.LayerNorm(d_seq).to(device)

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
        """Standard symmetric in-batch InfoNCE.

        The old ``-log(pos / sum(neg))`` objective could become negative and
        omitted the positive term from the softmax denominator.  Cross entropy
        over the full similarity matrix is non-negative and includes the
        matching pair in the denominator.
        """
        if z1.size(0) < 2:
            return z1.sum() * 0.0
        z1, z2 = self._normalize(z1), self._normalize(z2)
        logits = (z1 @ z2.T) / float(tau)
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (
            F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.T, labels)
        )

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
        # reps: [seq, graph, 3d]，形状应一致（默认 d_seq=d_graph=d_3d）
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
                return_contrastive=False,
                return_features=False,
                contrastive_mask: Optional[torch.Tensor] = None):

        B = p1_w2v.size(0)
        batch_device = p1_w2v.device

        # ---- 化合物 1D ----
        if self.ablate.use_cmp1d:
            cls_emb = self._encode_smiles_cls(cmp_smiles)
            def _cmp_1d_block(x): return self.seq_proj(x)
            cmp1d_in = cls_emb.detach()
            if self.training: cmp1d_in = cmp1d_in.requires_grad_(True)
            c_seq = self._ckpt('CP1', _cmp_1d_block, cmp1d_in)   # [B, d_seq]
        else:
            c_seq = torch.zeros(B, self.d_seq, device=batch_device)

        # ---- 化合物 2D 图 ----
        if self.ablate.use_cmp_graph:
            def _cmp_2d_block(anchor):
                raw = self.cmp_gnn(cmp_g)
                gemb = self._ensure_graph_tensor(raw)
                return gemb + anchor.sum() * 0
            anchor = torch.ones(1, device=device, requires_grad=True) if self.training else torch.ones(1, device=device)
            c_graph = self._ckpt('CP2', _cmp_2d_block, anchor)  # [B, d_graph]
        else:
            c_graph = torch.zeros(B, self.d_graph, device=batch_device)

        # ---- Uni-Mol2 compound embedding ----
        if self.ablate.use_cmp_3d:
            def _cmp_3d_block(x): return self.cmp_3d_lin(x)
            cmp_3d_in = cmp_3d.squeeze(1)
            if self.training: cmp_3d_in = cmp_3d_in.detach().requires_grad_(True)
            c_3d = self._ckpt('CP3', _cmp_3d_block, cmp_3d_in)  # [B, d_3d]
        else:
            c_3d = torch.zeros(B, self.d_3d, device=batch_device)

        # ---- 蛋白 1D ----
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
            p1_seq = torch.zeros(B, self.d_seq, device=batch_device)
            p2_seq = torch.zeros(B, self.d_seq, device=batch_device)

        # ---- 蛋白 2D 图 ----
        if self.ablate.use_prot_graph:
            def _prot_2d_block(anchor, g):
                raw = self.prot_gnn(g)
                ge = self._ensure_graph_tensor(raw)
                return ge + anchor.sum() * 0
            p_anchor = torch.ones(1, device=device, requires_grad=True) if self.training else torch.ones(1, device=device)
            p1_graph = self._ckpt('CP5', lambda a: _prot_2d_block(a, p1_g), p_anchor)
            p2_graph = self._ckpt('CP5', lambda a: _prot_2d_block(a, p2_g), p_anchor)
        else:
            p1_graph = torch.zeros(B, self.d_graph, device=batch_device)
            p2_graph = torch.zeros(B, self.d_graph, device=batch_device)

        # ---- ESM-2 protein sequence embedding (legacy field: use_prot_3d) ----
        if self.ablate.use_prot_3d:
            def _prot_3d_block(x): return self.prot_3d_lin(x)
            p1_3d_in = p1_3d.mean(dim=1); p2_3d_in = p2_3d.mean(dim=1)
            if self.training:
                p1_3d_in = p1_3d_in.detach().requires_grad_(True)
                p2_3d_in = p2_3d_in.detach().requires_grad_(True)
            p1_3d_proj = self._ckpt('CP6', _prot_3d_block, p1_3d_in)
            p2_3d_proj = self._ckpt('CP6', _prot_3d_block, p2_3d_in)
        else:
            p1_3d_proj = torch.zeros(B, self.d_3d, device=batch_device)
            p2_3d_proj = torch.zeros(B, self.d_3d, device=batch_device)

        if self.ablate.fusion_v2:
            def _v2_fuse(tokens, enabled):
                # tokens: [B, 3, d], enabled: length-3 booleans.  Disabled
                # modalities are masked from both attention and pooling.
                mask = torch.tensor(
                    [not bool(x) for x in enabled], dtype=torch.bool, device=tokens.device
                ).unsqueeze(0).expand(tokens.size(0), -1)
                if bool(mask.all(dim=1).any()):
                    raise RuntimeError("v2 fusion requires at least one enabled modality")
                # A disabled modality must be causally absent, rather than a
                # zero/constant token that can still pass through attention.
                tokens = tokens.masked_fill(mask.unsqueeze(-1), 0.0)
                if self.ablate.use_intra_modal_attention_v2:
                    tokens = self.v2_modal_encoder(tokens, src_key_padding_mask=mask)
                    tokens = tokens.masked_fill(mask.unsqueeze(-1), 0.0)
                if self.ablate.use_sample_dynamic_weights_v2:
                    scores = self.v2_modal_gate(tokens).squeeze(-1).masked_fill(mask, -1e9)
                    weights = torch.softmax(scores, dim=1)
                else:
                    weights = (~mask).float()
                    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
                return (tokens * weights.unsqueeze(-1)).sum(dim=1)

            v_c = _v2_fuse(
                torch.stack([c_seq, c_graph, c_3d], dim=1),
                [self.ablate.use_cmp1d, self.ablate.use_cmp_graph, self.ablate.use_cmp_3d],
            )
            protein_enabled = [
                self.ablate.use_prot1d, self.ablate.use_prot_graph, self.ablate.use_prot_3d
            ]
            v_p1 = _v2_fuse(torch.stack([p1_seq, p1_graph, p1_3d_proj], dim=1), protein_enabled)
            v_p2 = _v2_fuse(torch.stack([p2_seq, p2_graph, p2_3d_proj], dim=1), protein_enabled)
            if self.ablate.use_cross_object_attention_v2:
                objects = self.v2_object_encoder(torch.stack([v_c, v_p1, v_p2], dim=1))
                v_c, v_p1, v_p2 = objects[:, 0], objects[:, 1], objects[:, 2]
        else:
            # ---- Legacy IntraFusion + weighting ----
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

            # Legacy code performs object-local fusion; retained only for old
            # checkpoints and explicitly labelled as legacy in new reports.
            if self.ablate.use_cross_fuse:
                def _cross_block(vx):
                    out = self.cross_fuse(vx, vx, vx)
                    return out[0] if isinstance(out, (tuple, list)) else out
                v_c  = _cross_block(v_c)
                v_p1 = _cross_block(v_p1)
                v_p2 = _cross_block(v_p2)

        # ---- 归一化 + LN ----
        def _safe_embed(x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
            return norm(F.normalize(x, p=2, dim=-1, eps=1e-6))
        v_c  = _safe_embed(v_c, self.compound_object_norm)
        v_p1 = _safe_embed(v_p1, self.protein_object_norm)
        v_p2 = _safe_embed(v_p2, self.protein_object_norm)

        # ---- Hybrid 头（按开关拼接）----
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
        if self.ablate.head_use_hadamards:
            parts.append(torch.cat([(v_c * v_p1),
                                    (v_c * v_p2),
                                    (v_p1 * v_p2)], dim=-1))        # [B,3d]
        if not parts:
            parts.append(torch.cat([v_c, v_p1, v_p2], dim=-1))       # 兜底
        hybrid_feat_vec = torch.cat(parts, dim=-1).float()

        feat = self.hadamard_proj(hybrid_feat_vec) if self.hadamard_proj is not None else hybrid_feat_vec
        logits = self.classifier(feat).squeeze(-1)

        if not return_contrastive:
            if return_features:
                return logits, feat
            return logits

        if self.ablate.use_contrastive:
            nce_c, nce_p1, nce_p2 = v_c, v_p1, v_p2
            if contrastive_mask is not None:
                mask = contrastive_mask.reshape(-1).to(device=v_c.device, dtype=torch.bool)
                nce_c, nce_p1, nce_p2 = v_c[mask], v_p1[mask], v_p2[mask]
            closs = (
                self.info_nce_loss(nce_c, nce_p1) +
                self.info_nce_loss(nce_c, nce_p2) +
                self.info_nce_loss(nce_p1, nce_p2)
            ) / 3.0
        else:
            closs = torch.tensor(0.0, device=logits.device)

        if return_features:
            return logits, closs, feat
        return logits, closs

    # ====== MoLFormer 编码 + 缓存（保留原逻辑）======
    def _encode_smiles_cls(self, smiles_list: Union[str, List[str]]):
        if isinstance(smiles_list, (list, tuple)): smi_list = list(smiles_list)
        else: smi_list = [smiles_list]
        cached, to_compute_idx, to_compute_smi = [None]*len(smi_list), [], []
        if self.enable_smiles_cache and self.smiles_cache is not None:
            for i, s in enumerate(smi_list):
                if s in self.smiles_cache: cached[i] = self.smiles_cache[s]
                else: to_compute_idx.append(i); to_compute_smi.append(s)
        else:
            to_compute_idx = list(range(len(smi_list))); to_compute_smi = smi_list
        if to_compute_smi:
            tokens = self.tokenizer(to_compute_smi, return_tensors="pt", padding=True, truncation=True)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            mask_rate = getattr(self, 'smiles_mask_rate', 0.0) if self.training else 0.0
            if mask_rate:
                ids = tokens['input_ids'].clone()
                eligible = tokens['attention_mask'].bool()
                for special_id in self.tokenizer.all_special_ids:
                    eligible &= ids.ne(special_id)
                selected = (torch.rand(ids.shape, device=ids.device) < mask_rate) & eligible
                action = torch.rand(ids.shape, device=ids.device)
                ids[selected & (action < 0.8)] = self.tokenizer.mask_token_id
                random_positions = selected & (action >= 0.8) & (action < 0.9)
                ids[random_positions] = torch.randint(len(self.tokenizer),
                    (int(random_positions.sum()),), device=ids.device)
                tokens['input_ids'] = ids
            with torch.no_grad():
                out = self.molf(**tokens, return_dict=True)
                cls_part = out.last_hidden_state[:, 0, :].detach()
            for j, idx in enumerate(to_compute_idx):
                if self.enable_smiles_cache and self.smiles_cache is not None:
                    self.smiles_cache[to_compute_smi[j]] = cls_part[j].cpu()
                cached[idx] = cls_part[j]
        cls_emb = torch.stack([t if t.device == device else t.to(device) for t in cached], dim=0)
        return cls_emb
