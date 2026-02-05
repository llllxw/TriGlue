import torch, dgl
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, GINConv,
    global_max_pool as gmp,
    global_add_pool
)
from torch.nn import Sequential, Linear, ReLU
from dgl import mean_nodes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 工具：把 batched‑DGL 节点特征重排成 [B, Nmax, ...] ----------
def dgl_split(bg: dgl.DGLGraph, feats: torch.Tensor) -> torch.Tensor:
    """把 batched 图节点张量 v-> [B, Nmax, C]（pad 0）"""
    nums = bg.batch_num_nodes()                     # [B]
    max_n = int(nums.max())
    batch_idx = torch.repeat_interleave(
        torch.arange(bg.batch_size, device=bg.device), nums
    )                                               # [∑ni]
    base = torch.cat([nums.new_zeros(1), nums.cumsum(0)])[:-1]
    idx = torch.arange(bg.num_nodes(), device=bg.device) - base[batch_idx] + batch_idx * max_n
    out = feats.new_zeros(bg.batch_size * max_n, *feats.shape[1:])
    out[idx] = feats
    return out.view(bg.batch_size, max_n, *feats.shape[1:])

# ---------- 1. GCN ---------------------------------------------------------
class GCNNet(nn.Module):
    def __init__(self, num_feat: int, out_dim: int = 128, dropout: float = .2):
        super().__init__()
        self.conv1 = GCNConv(num_feat, num_feat)
        self.conv2 = GCNConv(num_feat, num_feat * 2)
        self.conv3 = GCNConv(num_feat * 2, num_feat * 4)
        self.fc1   = nn.Linear(num_feat * 4, 1024)
        self.fc2   = nn.Linear(1024, out_dim)
        self.act, self.dp = nn.ReLU(), nn.Dropout(dropout)

    def forward(self, g: dgl.DGLGraph):
        x = g.ndata["feats"].float().to(device)
        src, dst = g.edges()
        edge_index = torch.stack([src, dst], dim=1).T.long().to(device)
        batch = g.ndata["graph_id"].long().to(device)

        for conv in (self.conv1, self.conv2, self.conv3):
            x = self.act(conv(x, edge_index))
        node_repr = self.dp(self.fc2(self.act(self.fc1(dgl_split(g, x)))))   # [B, Nmax, out_dim]

        graph_repr = self.dp(self.fc2(self.act(self.fc1(gmp(x, batch)))))    # [B, out_dim]
        return node_repr, graph_repr

# ---------- 2. GAT ---------------------------------------------------------
class GATNet(nn.Module):
    def __init__(self, num_feat: int, out_dim: int = 128, heads: int = 10, dropout: float = .2):
        super().__init__()
        self.gat1 = GATConv(num_feat, num_feat, heads=heads, dropout=dropout)
        self.gat2 = GATConv(num_feat * heads, out_dim, dropout=dropout)
        self.fc   = nn.Linear(out_dim, out_dim)
        self.act, self.dp = nn.ReLU(), nn.Dropout(dropout)

    def forward(self, g):
        x = g.ndata["feats"].float().to(device)
        src, dst = g.edges()
        edge_index = torch.stack([src, dst], dim=1).T.long().to(device)
        batch = g.ndata["graph_id"].long().to(device)

        x = self.act(self.gat1(x, edge_index))
        x = self.act(self.gat2(x, edge_index))

        node_repr  = self.fc(dgl_split(g, x))
        graph_repr = self.fc(gmp(x, batch))
        return node_repr, graph_repr

# ---------- 3. GIN ---------------------------------------------------------
class GINConvNet(nn.Module):
    def __init__(self, num_feat: int, out_dim: int = 128, dim: int = 32, dropout: float = .2):
        super().__init__()
        def mlp():
            return Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(Sequential(Linear(num_feat, dim), ReLU(), Linear(dim, dim)))
        self.bn1   = nn.BatchNorm1d(dim)
        self.conv2, self.bn2 = GINConv(mlp()), nn.BatchNorm1d(dim)
        self.conv3, self.bn3 = GINConv(mlp()), nn.BatchNorm1d(dim)
        self.conv4, self.bn4 = GINConv(mlp()), nn.BatchNorm1d(dim)
        self.conv5, self.bn5 = GINConv(mlp()), nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, out_dim)
        self.act, self.dp = nn.ReLU(), nn.Dropout(dropout)

    def forward(self, g):
        x = g.ndata["feats"].float().to(device)
        src, dst = g.edges()
        edge_index = torch.stack([src, dst], dim=1).T.long().to(device)
        batch = g.ndata["graph_id"].long().to(device)

        for conv, bn in ((self.conv1, self.bn1),
                         (self.conv2, self.bn2),
                         (self.conv3, self.bn3),
                         (self.conv4, self.bn4),
                         (self.conv5, self.bn5)):
            x = self.act(bn(conv(x, edge_index)))

        graph_repr = self.dp(self.act(self.fc1(global_add_pool(x, batch))))
        return graph_repr            # GIN 仅返回图级向量


# -------- Multi-head attention on DGL edges --------
class _MHAEdge(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads):
        super().__init__()
        assert out_dim % n_heads == 0, "out_dim 必须能被 n_heads 整除"
        self.h = n_heads
        self.dk = out_dim // n_heads
        self.Wq = nn.Linear(in_dim, out_dim, bias=False)
        self.Wk = nn.Linear(in_dim, out_dim, bias=False)
        self.Wv = nn.Linear(in_dim, out_dim, bias=False)
        self.We = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, g: dgl.DGLGraph, h, e):
        # h: [N, Din] -> [N, H, dk]
        Q = self.Wq(h).view(-1, self.h, self.dk)
        K = self.Wk(h).view(-1, self.h, self.dk)
        V = self.Wv(h).view(-1, self.h, self.dk)
        # e: [E, Ein] -> [E, H, dk]
        E = self.We(e).view(-1, self.h, self.dk)

        with g.local_scope():
            g.ndata['Q'] = Q
            g.ndata['K'] = K
            g.ndata['V'] = V
            g.edata['E'] = E

            # score = (K_src · Q_dst) / sqrt(dk) ，再乘显式边权 E（逐 head 逐通道）
            g.apply_edges(lambda edges: {
                'score': (edges.src['K'] * edges.dst['Q']).sum(-1, keepdim=True) / math.sqrt(self.dk)
            })
            # 将 E 显式乘进 score
            g.edata['score'] = g.edata['score'] * (1.0 + g.edata['E'])

            # softmax（按入边）
            g.update_all(fn.copy_e('score', 'a'), fn.sum('a', 'den'))
            den = g.ndata['den'].clamp_min_(1e-6)  # [N, H, 1]
            # 归一后作为权重，再聚合 V
            g.apply_edges(lambda edges: {'alpha': torch.exp(edges.data['score'])})
            g.update_all(fn.copy_e('alpha', 'num'), fn.sum('num', 'Z'))
            Z = g.ndata['Z'].clamp_min_(1e-6)

            # 消息：alpha * V_src
            g.apply_edges(lambda edges: {'m': edges.src['V'] * edges.data['alpha']})
            g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'h_out'))  # [N, H, dk]

            h_out = g.ndata['h_out'] / Z            # [N, H, dk]
            h_out = h_out.reshape(-1, self.h * self.dk)  # [N, H*dk]
            # 可把边的注意力导回去（此处不需要）
            return h_out

# -------- Transformer layer on graphs --------
class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.1, residual=True):
        super().__init__()
        self.residual = residual
        self.attn = _MHAEdge(in_dim, out_dim, n_heads)
        self.lin   = nn.Linear(out_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.norm2 = nn.LayerNorm(out_dim)
        self.dp    = nn.Dropout(dropout)

    def forward(self, g, h, e):
        h_attn = self.attn(g, h, e)
        h = h + self.dp(self.lin(h_attn)) if self.residual else self.dp(self.lin(h_attn))
        h = self.norm1(h)
        h2 = self.dp(self.ffn(h))
        h = h + h2 if self.residual else h2
        h = self.norm2(h)
        return h

# -------- GraphTransformer encoder (graph-level embedding) --------
class GraphTransformer(nn.Module):
    """
    node_dim: 节点特征维（如 44 / 41）
    edge_dim: 边特征维（如无，则传 1；内部会在缺省时用全 0 占位）
    out_dim : 输出图级表征维
    """
    def __init__(self, node_dim, edge_dim=1, hidden_dim=128, out_dim=128,
                 n_layers=3, n_heads=4, in_dp=0.1, dropout=0.1, pos_enc_dim=8):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim 必须整除 n_heads"
        self.edge_dim = edge_dim
        self.pos_dim  = pos_enc_dim

        self.lin_node = nn.Linear(node_dim, hidden_dim)
        self.lin_pos  = nn.Linear(pos_enc_dim, hidden_dim)
        self.lin_edge = nn.Linear(edge_dim, hidden_dim)

        self.in_dp = nn.Dropout(in_dp)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, hidden_dim, n_heads=n_heads, dropout=dropout, residual=True)
            for _ in range(n_layers - 1)
        ])
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, n_heads=n_heads, dropout=dropout, residual=True))

        self.readout = nn.LayerNorm(out_dim)

    def forward(self, g: dgl.DGLGraph):
        h = g.ndata['feats'].float()
        # 位置编码可选
        if 'lap_pos_enc' in g.ndata:
            p = g.ndata['lap_pos_enc'].float()
            # 若维度不足则补零到 pos_dim
            if p.size(-1) != self.pos_dim:
                if p.size(-1) < self.pos_dim:
                    pad = torch.zeros(p.size(0), self.pos_dim - p.size(-1), device=p.device, dtype=p.dtype)
                    p = torch.cat([p, pad], dim=-1)
                else:
                    p = p[:, :self.pos_dim]
        else:
            p = torch.zeros(h.size(0), self.pos_dim, device=h.device, dtype=h.dtype)

        if 'feats' in g.edata:
            e = g.edata['feats'].float()
            if e.size(-1) != self.edge_dim:
                # 边维不匹配就线性截断/补零到 edge_dim
                if e.size(-1) < self.edge_dim:
                    pad = torch.zeros(e.size(0), self.edge_dim - e.size(-1), device=e.device, dtype=e.dtype)
                    e = torch.cat([e, pad], dim=-1)
                else:
                    e = e[:, :self.edge_dim]
        else:
            e = torch.zeros(g.num_edges(), self.edge_dim, device=h.device, dtype=h.dtype)

        h = self.in_dp(self.lin_node(h) + self.lin_pos(p))
        e = self.lin_edge(e)

        for layer in self.layers:
            h = layer(g, h, e)

        g.ndata['h_gt'] = h
        # 图级读出：mean_nodes
        graph_repr = dgl.mean_nodes(g, 'h_gt')   # [B, out_dim]
        return graph_repr