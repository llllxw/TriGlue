import math, torch, torch.nn as nn, torch.nn.functional as F, dgl
import dgl.function as fn
import numpy as np

# ========= 辅助函数 =========
def src_dot_dst(s, t, out):          # <src,h> · <dst,h>
    return lambda edges: {out: (edges.src[s] * edges.dst[t])}

def edge_scale(f, c):                # /√d_k
    return lambda edges: {f: edges.data[f] / c}

def imp_exp_attn(score_key, edge_key):   # 乘显式边权
    return lambda edges: {score_key: edges.data[score_key] * edges.data[edge_key]}

def copy_edge(feat):                 # 把 attention 分数存给 FFN_e
    return lambda edges: {'e_out': edges.data[feat]}

def exp_clamp(field):                # softmax 前 clamp
    return lambda edges: {field: torch.exp(edges.data[field].sum(-1, keepdim=True).clamp(-5, 5))}

# ========= Multi‑Head Attention =========
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads, bias=False):
        super().__init__()
        self.h = n_heads
        self.d_k = out_dim
        self.W_Q = nn.Linear(in_dim, out_dim * n_heads, bias=bias)
        self.W_K = nn.Linear(in_dim, out_dim * n_heads, bias=bias)
        self.W_V = nn.Linear(in_dim, out_dim * n_heads, bias=bias)
        self.W_E = nn.Linear(in_dim, out_dim * n_heads, bias=bias)

    def forward(self, g: dgl.DGLGraph, h, e):
        g = g.local_var()                        # 安全拷贝
        g.ndata['Q_h'] = self.W_Q(h).view(-1, self.h, self.d_k)
        g.ndata['K_h'] = self.W_K(h).view(-1, self.h, self.d_k)
        g.ndata['V_h'] = self.W_V(h).view(-1, self.h, self.d_k)
        g.edata['E_h'] = self.W_E(e).view(-1, self.h, self.d_k)

        # 注意力分数
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))
        g.apply_edges(edge_scale('score', math.sqrt(self.d_k)))
        g.apply_edges(imp_exp_attn('score', 'E_h'))
        g.apply_edges(copy_edge('score'))
        g.apply_edges(exp_clamp('score'))

        # message‑passing
        eid = g.edges()
        g.send_and_recv(eid,
                        fn.u_mul_e('V_h', 'score', 'm'),
                        fn.sum('m', 'wV'))
        g.send_and_recv(eid,
                        fn.copy_e('score', 's'),
                        fn.sum('s', 'z'))

        h_out = g.ndata['wV'] / (g.ndata['z'] + 1e-6)
        e_out = g.edata['e_out']
        return h_out.view(-1, self.h * self.d_k), e_out.view(-1, self.h * self.d_k)

# ========= Transformer‑Layer =========
class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.1,
                 layer_norm=True, batch_norm=False, residual=True, bias=False):
        super().__init__()
        self.residual = residual
        self.attn = MultiHeadAttentionLayer(in_dim, out_dim // n_heads, n_heads, bias)

        self.lin_h = nn.Linear(out_dim, out_dim)
        self.lin_e = nn.Linear(out_dim, out_dim)

        self.norm1_h = nn.LayerNorm(out_dim) if layer_norm else None
        self.norm1_e = nn.LayerNorm(out_dim) if layer_norm else None
        self.bn1_h   = nn.BatchNorm1d(out_dim) if batch_norm else None
        self.bn1_e   = nn.BatchNorm1d(out_dim) if batch_norm else None

        self.ff_h = nn.Sequential(nn.Linear(out_dim, 2*out_dim),
                                  nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(2*out_dim, out_dim))
        self.ff_e = nn.Sequential(nn.Linear(out_dim, 2*out_dim),
                                  nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(2*out_dim, out_dim))

        self.norm2_h = nn.LayerNorm(out_dim) if layer_norm else None
        self.norm2_e = nn.LayerNorm(out_dim) if layer_norm else None
        self.bn2_h   = nn.BatchNorm1d(out_dim) if batch_norm else None
        self.bn2_e   = nn.BatchNorm1d(out_dim) if batch_norm else None
        self.dp = nn.Dropout(dropout)

    def forward(self, g, h, e):
        # --- Multi‑head attention ---
        h1, e1 = self.attn(g, h, e)
        h2 = self.dp(self.lin_h(h1))
        e2 = self.dp(self.lin_e(e1))
        if self.residual:
            h = h + h2
            e = e + e2
        if self.norm1_h: h = self.norm1_h(h)
        if self.norm1_e: e = self.norm1_e(e)
        if self.bn1_h  : h = self.bn1_h(h)
        if self.bn1_e  : e = self.bn1_e(e)

        # --- Feed‑Forward ---
        h3 = self.dp(self.ff_h(h))
        e3 = self.dp(self.ff_e(e))
        if self.residual:
            h = h + h3
            e = e + e3
        if self.norm2_h: h = self.norm2_h(h)
        if self.norm2_e: e = self.norm2_e(e)
        if self.bn2_h  : h = self.bn2_h(h)
        if self.bn2_e  : e = self.bn2_e(e)
        return h, e
