# file: graph_transformer_noedge.py
import math, torch, torch.nn as nn, torch.nn.functional as F
import dgl, dgl.function as fn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------- util --------
def src_dot_dst(src_field, dst_field, out_field):
    def _fn(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
    return _fn

def scaled_exp(field, d):
    def _fn(edges):
        return {field: torch.exp((edges.data[field] / d).clamp(-5, 5))}
    return _fn

# -------- attention head (no edge feats) --------
class _MHA(nn.Module):
    def __init__(self, in_dim, out_dim, heads, bias=False):
        super().__init__()
        self.h, self.d = heads, out_dim
        self.Q = nn.Linear(in_dim, out_dim * heads, bias=bias)
        self.K = nn.Linear(in_dim, out_dim * heads, bias=bias)
        self.V = nn.Linear(in_dim, out_dim * heads, bias=bias)

    def propagate(self, g):
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))
        g.apply_edges(scaled_exp('score', math.sqrt(self.d)))

        eid = g.edges()
        g.send_and_recv(eid, fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eid, fn.copy_e('score', 'score'),     fn.sum('score', 'z'))

    def forward(self, g, h):
        g = g.local_var()
        g.ndata['Q_h'] = self.Q(h).view(-1, self.h, self.d)
        g.ndata['K_h'] = self.K(h).view(-1, self.h, self.d)
        g.ndata['V_h'] = self.V(h).view(-1, self.h, self.d)

        self.propagate(g)
        out = g.ndata['wV'] / (g.ndata['z'] + 1e-6)
        return out.view(-1, self.h * self.d)

# -------- GT layer (no edge feats) --------
class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1,
                 layer_norm=True, batch_norm=True, residual=True, bias=False):
        super().__init__()
        self.residual, self.dropout = residual, dropout
        self.attn  = _MHA(in_dim, out_dim // heads, heads, bias)
        self.O     = nn.Linear(out_dim, out_dim)
        self.FFN1  = nn.Linear(out_dim, out_dim * 2)
        self.FFN2  = nn.Linear(out_dim * 2, out_dim)

        self.ln1 = nn.LayerNorm(out_dim) if layer_norm else nn.Identity()
        self.ln2 = nn.LayerNorm(out_dim) if layer_norm else nn.Identity()
        self.bn1 = nn.BatchNorm1d(out_dim) if batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm1d(out_dim) if batch_norm else nn.Identity()

    def forward(self, g, h):
        h0 = h
        h  = self.attn(g, h)
        h  = F.dropout(self.O(h), self.dropout, self.training)
        if self.residual: h = h + h0
        h = self.bn1(self.ln1(h))

        h1 = h
        h  = F.relu(self.FFN1(h))
        h  = F.dropout(self.FFN2(h), self.dropout, self.training)
        if self.residual: h = h + h1
        h = self.bn2(self.ln2(h))
        return h

# -------- small wrapper --------------- 
class GraphTransformer(nn.Module):
    def __init__(self, node_dim, hidden=128, out_dim=128,
                 layers=3, heads=4, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(node_dim, hidden)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden, hidden, heads, dropout) for _ in range(layers-1)
        ] + [GraphTransformerLayer(hidden, out_dim, heads, dropout)])
    def forward(self, g):
        h = self.embed(g.ndata['feats'].float().to(device))
        for layer in self.layers:
            h = layer(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')      # 图级表示
