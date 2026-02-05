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

        #  [batch_szie, num_heads, seq_len, depth] = [3, 8, 5, 512/8=64]
        Q = Q.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        # [batch_szie, num_heads, seq_len, seq_len] = [3, 8, 5, 5]
        att_weights = torch.matmul(Q, K.transpose(-2, -1))
        att_weights = att_weights / math.sqrt(self.depth)

        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)

        if self.att_dropout > 0.0:
            att_weights = F.dropout(att_weights, p=self.att_dropout)
        output = torch.matmul(att_weights, V)
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


        # self.Wq = nn.Linear(emb_dim, emb_dim).to(device)
        # self.Wk = nn.Linear(emb_dim, emb_dim).to(device)
        # self.Wv = nn.Linear(emb_dim, emb_dim).to(device)
        self.Wq = nn.Linear(emb_dim, emb_dim, bias=False)
        self.Wk = nn.Linear(emb_dim, emb_dim, bias=False)
        self.Wv = nn.Linear(emb_dim, emb_dim, bias=False)



    def forward(self, x, context, pad_mask=None):

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

        return vx, vy  


class TriModalFusion(nn.Module):

    def __init__(self,
                 d_emb: int,
                 heads: int = 8,
                 pf_ratio: int = 4,
                 dropout: float = 0.1,
                 n_layers: int = 2):
        super().__init__()
        self.f_sg = CrossFusion(d_emb, heads, pf_ratio, dropout, n_layers)
        self.f_g3 = CrossFusion(d_emb, heads, pf_ratio, dropout, n_layers)
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

        v_seq_fused   = seq_sg + seq_3s       # [B, d_emb]
        v_graph_fused = graph_sg + graph_g3   # [B, d_emb]
        v_3d_fused    = d3_g3   + d3_3s       # [B, d_emb]

        return v_seq_fused, v_graph_fused, v_3d_fused

