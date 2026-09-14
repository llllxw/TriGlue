import math
import torch.nn.functional as F
import torch.nn as nn
import torch
from typing import Tuple
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, att_dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.att_dropout = att_dropout

        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.depth = emb_dim // num_heads

        # self.Wq = nn.Linear(emb_dim, emb_dim, bias=False).to(device)
        # self.Wk = nn.Linear(emb_dim, emb_dim, bias=False).to(device)
        # self.Wv = nn.Linear(emb_dim, emb_dim, bias=False).to(device)
        self.Wq = nn.Linear(emb_dim, emb_dim, bias=False)
        self.Wk = nn.Linear(emb_dim, emb_dim, bias=False)
        self.Wv = nn.Linear(emb_dim, emb_dim, bias=False)

        # self.fc = nn.Linear(emb_dim, emb_dim).to(device)
        self.fc = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, pad_mask=None):
        # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
        batch_size = x.size(0)

        # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # 分头 [batch_szie, num_heads, seq_len, depth] = [3, 8, 5, 512/8=64]
        Q = Q.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        # [batch_szie, num_heads, seq_len, seq_len] = [3, 8, 5, 5]
        att_weights = torch.matmul(Q, K.transpose(-2, -1))
        att_weights = att_weights / math.sqrt(self.depth)

        if pad_mask is not None:
            # 因为是多头，所以mask矩阵维度要扩充到4维  [batch_size, seq_len, seq_len] -> [batch_size, nums_head, seq_len, seq_len]
            pad_mask = pad_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)

        # 自己的多头注意力效果没有torch的好，我猜是因为它的dropout给了att权重，而不是fc
        if self.att_dropout > 0.0:
            att_weights = F.dropout(att_weights, p=self.att_dropout, training=self.training)

        # [batch_szie, num_heads, seq_len, depth] = [3, 8, 5, 64]
        output = torch.matmul(att_weights, V)

        # 不同头的结果拼接 [batch_szie, seq_len, emb_dim] = [3, 5, 512]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.emb_dim)

        output = self.fc(output)
        return output

class CrossMultiAttention(nn.Module):
    def __init__(self,emb_dim, num_heads, att_dropout=0.0, aropout=0.0):
        super(CrossMultiAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.scale = emb_dim ** -0.5

        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.depth = emb_dim // num_heads


        #self.proj_in = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)

        # self.Wq = nn.Linear(emb_dim, emb_dim).to(device)
        # self.Wk = nn.Linear(emb_dim, emb_dim).to(device)
        # self.Wv = nn.Linear(emb_dim, emb_dim).to(device)
        self.Wq = nn.Linear(emb_dim, emb_dim, bias=False)
        self.Wk = nn.Linear(emb_dim, emb_dim, bias=False)
        self.Wv = nn.Linear(emb_dim, emb_dim, bias=False)

        #self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x, context, pad_mask=None):
        '''

        :param x: [batch_size,seq_len,emb_dim]
        :param context: [batch_szie, seq_len, emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        batch_size= x.shape[0]
        Q = self.Wq(x)  # [batch_size, seq_len, emb_dim] = [3, 262144, 512]
        K = self.Wk(context)  # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
        V = self.Wv(context)

        Q = Q.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, h*w, depth]
        K = K.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, seq_len, depth]
        V = V.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        # [batch_size, num_heads, h*w, seq_len]
        att_weights = torch.einsum('bnid,bnjd -> bnij', Q, K)
        att_weights = att_weights * self.scale

        if pad_mask is not None:
            # 因为是多头，所以mask矩阵维度要扩充到4维  [batch_size, h*w, seq_len] -> [batch_size, nums_head, h*w, seq_len]
            pad_mask = pad_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        out = torch.einsum('bnij, bnjd -> bnid', att_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.emb_dim)   # [batch_size, seq_len, emb_dim]
        #输出的结果是x的seq_len!
        #print(out.shape)
        return out
class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.pf_dim = pf_dim
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1).to(device)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1).to(device)  # convolution neural units
        self.do = nn.Dropout(dropout).to(device)
    def forward(self, x):
        # x = [batch size, sent len, hid dim]
        x = x.permute(0, 2, 1)
        # x = [batch size, hid dim, sent len]
        x = self.do(F.relu(self.fc_1(x)))
        # x = [batch size, pf dim, sent len]
        x = self.fc_2(x)
        # x = [batch size, hid dim, sent len]
        x = x.permute(0, 2, 1)
        # x = [batch size, sent len, hid dim]
        return x
class EncoderLayer_Self(nn.Module):
    def __init__(self,emd_dim,n_heads,pf_dim,dropout,multiheadattention,positionwisefeedforward):
        super().__init__()
        self.ln = nn.LayerNorm(emd_dim).to(device)
        self.sa = multiheadattention(emd_dim, n_heads, dropout)
        self.ea = multiheadattention(emd_dim, n_heads)
        self.pf = positionwisefeedforward(emd_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)
    def forward(self, trg,trg_mask=None):
        trg= self.sa(trg,trg_mask)
        trg = self.ln(trg + self.do(trg))
        trg= self.ea(trg, trg_mask)
        trg = self.ln(trg + self.do(trg))
        trg = self.ln(trg + self.do(self.pf(trg)))
        #输出维度为[batch_size,seq_len,hid_dim]
        return trg
class EncoderLayer_Cross(nn.Module):
    def __init__(self,emd_dim,n_heads,pf_dim,dropout,crossmultiattention,positionwisefeedforward):
        super().__init__()
        self.ln = nn.LayerNorm(emd_dim)
        self.sa = crossmultiattention(emd_dim, n_heads, dropout)
        self.ea = crossmultiattention(emd_dim, n_heads)
        self.pf = positionwisefeedforward(emd_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)
    def forward(self,trg,src,trg_mask=None,src_mask=None):
        trg = self.sa(trg,trg,trg_mask)

        trg = self.ln(trg + self.do(trg))

        trg = self.ea(trg, src, src_mask)

        trg = self.ln(trg + self.do(trg))

        trg = self.ln(trg + self.do(self.pf(trg)))
        #输出维度为[batch_size,seq_len,hid_dim]
        return trg

class IntraFusion(nn.Module):
    """单模态自注意力池化   x:[B,S,d]  →  v:[B,d]"""
    def __init__(self, d_emb=128, n_layers=2, heads=8, pf_ratio=4, dp=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                MultiHeadAttention(d_emb, heads, dp),
                PositionwiseFeedforward(d_emb, pf_ratio*d_emb, dp),
                nn.LayerNorm(d_emb), nn.LayerNorm(d_emb),
                nn.Dropout(dp)
            ]) for _ in range(n_layers)
        ])

    def forward(self, x, mask=None):
        for attn, ffn, ln1, ln2, drop in self.blocks:
            x = ln1(x + drop(attn(x, mask)))
            x = ln2(x + drop(ffn(x)))
        # mean‑pool (忽略 padding)
        if mask is not None:
            lens = (~mask.squeeze(1)).sum(1, keepdim=True)
            x = (x * (~mask).float()).sum(1) / lens
        else:
            x = x.mean(1)
        return x              # [B,d_emb]


class CrossFusion(nn.Module):
    """
    两模态交叉注意力融合：
      输入 x ∈ ℝ^{B×Sx×d}, y ∈ ℝ^{B×Sy×d}
      先让 x 针对 y 做 cross‐attention，再让 y 针对 x 做 cross‐attention，
      重复 n_layers 层；最后对每个模态做 mean‐pool 得到 [B×d]。
    """
    def __init__(self,
                 d_emb: int = 128,
                 heads: int =8 ,
                 pf_ratio: int = 4,
                 dropout: float = 0.1,
                 n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer_Cross(
                emd_dim=d_emb,
                n_heads=heads,
                pf_dim=d_emb * pf_ratio,
                dropout=dropout,
                crossmultiattention=CrossMultiAttention,
                positionwisefeedforward=PositionwiseFeedforward
            )
            for _ in range(n_layers)
        ])

    def forward(self,
                x: torch.Tensor,           # [B, Sx, d_emb]
                y: torch.Tensor,           # [B, Sy, d_emb]
                x_mask: torch.BoolTensor = None,  # [B, 1, Sx] 或者 [B,Sx]，可选
                y_mask: torch.BoolTensor = None   # [B, 1, Sy] 或者 [B,Sy]，可选
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            # x queries y, then y queries x
            x = layer(x, y, trg_mask=x_mask, src_mask=y_mask)
            y = layer(y, x, trg_mask=y_mask, src_mask=x_mask)
        # 最后全局池化
        if x_mask is not None:
            len_x = (~x_mask.squeeze(1)).sum(-1, keepdim=True)
            vx = (x * (~x_mask).float()).sum(1) / len_x
        else:
            vx = x.mean(1)

        if y_mask is not None:
            len_y = (~y_mask.squeeze(1)).sum(-1, keepdim=True)
            vy = (y * (~y_mask).float()).sum(1) / len_y
        else:
            vy = y.mean(1)

        return vx, vy  # 都是 [B, d_emb]


class TriModalFusion(nn.Module):
    """
    三模态交叉融合：分别做 (seq↔graph)、(graph↔3d)、(3d↔seq) 的 CrossFusion，
    然后把每个分支对应的两次输出加起来得到三条最终向量。
    """
    def __init__(self,
                 d_emb: int,
                 heads: int = 8,
                 pf_ratio: int = 4,
                 dropout: float = 0.1,
                 n_layers: int = 2):
        super().__init__()
        # seq↔graph 融合
        self.f_sg = CrossFusion(d_emb, heads, pf_ratio, dropout, n_layers)
        # graph↔3d 融合
        self.f_g3 = CrossFusion(d_emb, heads, pf_ratio, dropout, n_layers)
        # 3d↔seq 融合
        self.f_3s = CrossFusion(d_emb, heads, pf_ratio, dropout, n_layers)

    def forward(self,
                v_seq: torch.Tensor,   # [B, L_seq, d_emb]
                v_graph: torch.Tensor, # [B, L_graph, d_emb]
                v_3d: torch.Tensor,    # [B, L_3d, d_emb]
                seq_mask=None,
                graph_mask=None,
                d3_mask=None
    ):
        # seq vs graph
        seq_sg, graph_sg = self.f_sg(v_seq,   v_graph, seq_mask, graph_mask)
        # graph vs 3d
        graph_g3, d3_g3   = self.f_g3(v_graph, v_3d,   graph_mask, d3_mask)
        # 3d vs seq
        d3_3s, seq_3s     = self.f_3s(v_3d,    v_seq,   d3_mask, seq_mask)

        # 对每个模态，把两次融合结果相加：
        v_seq_fused   = seq_sg + seq_3s       # [B, d_emb]
        v_graph_fused = graph_sg + graph_g3   # [B, d_emb]
        v_3d_fused    = d3_g3   + d3_3s       # [B, d_emb]

        return v_seq_fused, v_graph_fused, v_3d_fused

###加入门控融合
# class TriModalFusion(nn.Module):
#     """
#     三模态交叉融合 + 门控：先做 seq↔graph、graph↔3d、3d↔seq 各自的 CrossFusion，
#     然后在每个分支内部用 gate 融合两次交叉结果。
#     """
#     def __init__(self, d_emb: int, heads: int = 8, pf_ratio: int = 4,
#                  dropout: float = 0.1, n_layers: int = 2):
#         super().__init__()
#         # ① 原有的三路 CrossFusion
#         self.f_sg = CrossFusion(d_emb, heads, pf_ratio, dropout, n_layers)
#         self.f_g3 = CrossFusion(d_emb, heads, pf_ratio, dropout, n_layers)
#         self.f_3s = CrossFusion(d_emb, heads, pf_ratio, dropout, n_layers)

#         # ② 门控网络（分别融合各分支两个输出），输出一个 [B,1] 的 gate
#         self.gate_seq   = nn.Sequential(nn.Linear(d_emb*2, 1), nn.Sigmoid())
#         self.gate_graph = nn.Sequential(nn.Linear(d_emb*2, 1), nn.Sigmoid())
#         self.gate_3d    = nn.Sequential(nn.Linear(d_emb*2, 1), nn.Sigmoid())

#     def forward(self, v_seq, v_graph, v_3d,
#                 seq_mask=None, graph_mask=None, d3_mask=None):
#         # 1) 三对 CrossFusion
#         seq_sg, graph_sg = self.f_sg(v_seq,   v_graph, seq_mask, graph_mask)
#         graph_g3, d3_g3  = self.f_g3(v_graph, v_3d,   graph_mask, d3_mask)
#         d3_3s, seq_3s    = self.f_3s(v_3d,    v_seq,   d3_mask, seq_mask)

#         # 2) 各分支内部门控融合
#         # — seq 分支 —
#         cat_s = torch.cat([seq_sg, seq_3s], dim=-1)           # [B,2*d_emb]
#         α_s   = self.gate_seq(cat_s)                          # [B,1]
#         v_seq_fused = α_s * seq_sg + (1-α_s) * seq_3s         # [B,d_emb]

#         # — graph 分支 —
#         cat_g = torch.cat([graph_sg, graph_g3], dim=-1)
#         α_g   = self.gate_graph(cat_g)
#         v_graph_fused = α_g * graph_sg + (1-α_g) * graph_g3

#         # — 3D 分支 —
#         cat_d = torch.cat([d3_g3, d3_3s], dim=-1)
#         α_d   = self.gate_3d(cat_d)
#         v_3d_fused    = α_d * d3_g3 + (1-α_d) * d3_3s

#         return v_seq_fused, v_graph_fused, v_3d_fused


#加入双向交叉注意力机制
# class TriModalFusion(nn.Module):
#     """
#     三模态交叉融合：在原有基础上加入双向CrossMultiAttention。
#     """
#     def __init__(self,
#                  d_emb: int,
#                  heads: int = 8,
#                  pf_ratio: int = 4,
#                  dropout: float = 0.1,
#                  n_layers: int = 2):
#         super().__init__()
#         # 保留原有CrossFusion实现
#         self.f_sg = CrossFusion(d_emb, heads, pf_ratio, dropout, n_layers)
#         self.f_g3 = CrossFusion(d_emb, heads, pf_ratio, dropout, n_layers)
#         self.f_3s = CrossFusion(d_emb, heads, pf_ratio, dropout, n_layers)
#         # 新增：双向CrossMultiAttention模块，用于更细粒度的双向交叉注意力
#         self.att_sg = CrossMultiAttention(d_emb, heads, att_dropout=dropout)
#         self.att_gs = CrossMultiAttention(d_emb, heads, att_dropout=dropout)
#         self.att_g3 = CrossMultiAttention(d_emb, heads, att_dropout=dropout)
#         self.att_3g = CrossMultiAttention(d_emb, heads, att_dropout=dropout)
#         self.att_3s = CrossMultiAttention(d_emb, heads, att_dropout=dropout)
#         self.att_s3 = CrossMultiAttention(d_emb, heads, att_dropout=dropout)

#     def forward(self, v_seq, v_graph, v_3d, seq_mask=None, graph_mask=None, d3_mask=None):
#         # 原有三对CrossFusion
#         seq_sg, graph_sg = self.f_sg(v_seq, v_graph, seq_mask, graph_mask)
#         graph_g3, d3_g3  = self.f_g3(v_graph, v_3d, graph_mask, d3_mask)
#         d3_3s, seq_3s    = self.f_3s(v_3d, v_seq, d3_mask, seq_mask)

#         # 新增：细粒度双向CrossMultiAttention
#         # seq <-> graph
#         seq2g = self.att_sg(v_seq.unsqueeze(1), v_graph.unsqueeze(1), pad_mask=graph_mask)
#         g2s   = self.att_gs(v_graph.unsqueeze(1), v_seq.unsqueeze(1), pad_mask=seq_mask)
#         seq2g = seq2g.squeeze(1)
#         g2s   = g2s.squeeze(1)
#         # graph <-> 3d
#         g2d3 = self.att_g3(v_graph.unsqueeze(1), v_3d.unsqueeze(1), pad_mask=d3_mask)
#         d32g = self.att_3g(v_3d.unsqueeze(1), v_graph.unsqueeze(1), pad_mask=graph_mask)
#         g2d3 = g2d3.squeeze(1)
#         d32g = d32g.squeeze(1)
#         # 3d <-> seq
#         d32s = self.att_3s(v_3d.unsqueeze(1), v_seq.unsqueeze(1), pad_mask=seq_mask)
#         s2d3 = self.att_s3(v_seq.unsqueeze(1), v_3d.unsqueeze(1), pad_mask=d3_mask)
#         d32s = d32s.squeeze(1)
#         s2d3 = s2d3.squeeze(1)

#         # 将原有融合结果与新增双向结果相加
#         v_seq_fused   = seq_sg + seq_3s + seq2g + s2d3
#         v_graph_fused = graph_sg + graph_g3 + g2s + d32g
#         v_3d_fused    = d3_g3 + d3_3s + g2d3 + d32s

#         return v_seq_fused, v_graph_fused, v_3d_fused
