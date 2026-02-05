import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class TextCNN(nn.Module):

    def __init__(self, in_dim: int, proj_out: int,
                 num_filters: int = 64, kernel_sizes=(3, 5, 7), dropout: float = 0.2):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_dim, out_channels=num_filters,
                      kernel_size=ks, padding=ks // 2)
            for ks in kernel_sizes
        ])
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), proj_out)
        self.norm = nn.LayerNorm(proj_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, V] -> [B, V, L]
        x = x.transpose(1, 2).contiguous().float()
        outs = []
        for conv in self.convs:
            z = self.act(conv(x))          # [B, C, L]
            z = torch.max(z, dim=2).values # Global max pool over L -> [B, C]
            outs.append(z)
        h = torch.cat(outs, dim=1)         # [B, C*K]
        h = self.dropout(h)
        h = self.fc(h)                     # [B, proj_out]
        return self.norm(h)

        
#RNN
class BILSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(BILSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=True)
        self.bigru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, bidirectional=True)
        self.birnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input):
        input=input.float()
        batch_size = input.size(0)
        seq_len = input.size(1)
        input = input.transpose(0, 1)

        hidden_state = torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim).to(input.device)
        cell_state = torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim).to(input.device)

        output, _ = self.bilstm(input, (hidden_state, cell_state))

        output = output.transpose(0, 1)

        glo = torch.mean(output, dim=1)

        output = self.fc(output)
        glo=self.fc(glo)
        return output,glo

#Transformer模型
class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)
    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = self.do(F.softmax(energy, dim=-1))

        x = torch.matmul(attention, V)

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        x = self.fc(x)

        return x
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask=None):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        说明：在encoder-decoder的Attention层中len_q(q1,..qt)和len_k(k1,...km)可能不同
        """
        d_k=Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / \
            np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]

        attn = nn.Softmax(dim=-1)(scores)  # 对最后一个维度(v)做softmax

        context = torch.matmul(attn, V)

        return context, attn
class MultiHeadAttention(nn.Module):


    def __init__(self,d_model,n_heads,dropout,device):
        super(MultiHeadAttention, self).__init__()
        self.d_model=d_model
        self.n_heads=n_heads
        self.device=device
        self.W_Q = nn.Linear(d_model, d_model * n_heads,
                             bias=False)  # q,k必须维度相同，不然无法做点积
        self.W_K = nn.Linear(d_model, d_model * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_model * n_heads, bias=False)

        self.fc = nn.Linear(n_heads * d_model, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask=None):
        residual, batch_size = input_Q, input_Q.size(0)

        Q = self.W_Q(input_Q).view(batch_size, -1,
                                   self.n_heads, self.d_model).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1,
                                   self.n_heads, self.d_model).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1,
                                   self.n_heads, self.d_model).transpose(1, 2)

        context, attn = ScaledDotProductAttention()(Q, K, V)

        context = context.transpose(1, 2).reshape(
            batch_size, -1, self.n_heads * self.d_model)

        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).to(self.device)(output + residual)

class Encoder(nn.Module):
    """protein feature extraction."""
    def __init__(self, protein_dim, hid_dim, n_layers,kernel_size , dropout, device):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)])   # convolutional layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim,self.hid_dim)

    def forward(self, protein):
        protein=protein.float()
        conv_input = self.fc(protein)
        conv_input = conv_input.permute(0, 2, 1)
        for i, conv in enumerate(self.convs):
            conved = conv(self.dropout(conv_input))
            conved = F.glu(conved, dim=1)
            conved = (conved + conv_input) * self.scale
            conv_input = conved

        conved = conved.permute(0,2,1)
        return conved

class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.pf_dim = pf_dim
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units
        self.do = nn.Dropout(dropout)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.do(F.relu(self.fc_1(x)))
        x = self.fc_2(x)
        x = x.permute(0, 2, 1)
        return x
class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):

        trg= self.sa(trg, trg, trg, trg_mask)

        trg = self.ln(trg + self.do(trg))

        trg= self.ea(trg, src, src, src_mask)

        trg = self.ln(trg + self.do(trg))

        trg = self.ln(trg + self.do(self.pf(trg)))
        return trg


class Decoder(nn.Module):
    """ compound feature extraction."""
    def __init__(self, atom_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = atom_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.layers = nn.ModuleList(
            [decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
             for _ in range(n_layers)])
        self.ft = nn.Linear(atom_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 2)

    def forward(self, trg, src, trg_mask=None,src_mask=None):
        trg = self.ft(trg)
        for layer in self.layers:
            trg= layer(trg, src,trg_mask,src_mask)
        """Use norm to determine which atom is significant. """
        norm = torch.norm(trg, dim=2)
        norm = F.softmax(norm, dim=1)
        sum = torch.zeros((trg.shape[0], self.hid_dim)).to(self.device)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                v = trg[i, j,]
                v = v * norm[i, j]
                sum[i,] += v
        return sum