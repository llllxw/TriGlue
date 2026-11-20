"""Single full-model reference implementation based on the final manuscript."""
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel,AutoTokenizer
import dgl
from dgl.nn.pytorch import GINConv

class TextCNN(nn.Module):
    """
    输入: x [B, L, V]  (V = one-hot 维)
    流程: Conv1d(in=V) -> GELU -> GlobalMaxPool(L) * 多核 -> concat -> Linear(proj_out)
    输出: [B, proj_out]
    """

    def __init__(self, in_dim: int, proj_out: int, num_filters: int=64, kernel_sizes=(3, 5, 7), dropout: float=0.2):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=in_dim, out_channels=num_filters, kernel_size=ks, padding=ks // 2) for ks in kernel_sizes])
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), proj_out)
        self.norm = nn.LayerNorm(proj_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        valid=x.abs().sum(-1).ne(0)
        x = x.transpose(1, 2).contiguous().float()
        outs = []
        for conv in self.convs:
            z = self.act(conv(x))
            z = z.masked_fill(~valid[:,None,:],-torch.inf).max(dim=2).values
            outs.append(z)
        h = torch.cat(outs, dim=1)
        h = self.dropout(h)
        h = self.fc(h)
        return self.norm(h)

class GINConvNet(nn.Module):
    def __init__(self,num_feat,out_dim=128,dim=32,dropout=.2,layers=5):
        super().__init__()
        self.convs=nn.ModuleList([
            GINConv(nn.Sequential(nn.Linear(num_feat if i==0 else dim,dim),
                                  nn.ReLU(),nn.Linear(dim,dim)),'sum')
            for i in range(layers)])
        self.norms=nn.ModuleList([nn.BatchNorm1d(dim) for _ in range(layers)])
        self.output=nn.Sequential(nn.Linear(dim,out_dim),nn.ReLU(),nn.Dropout(dropout))
    def forward(self,graph):
        x=graph.ndata['feats'].float()
        for conv,norm in zip(self.convs,self.norms):x=norm(conv(graph,x)).relu()
        with graph.local_scope():
            graph.ndata['_encoded']=x
            return self.output(dgl.sum_nodes(graph,'_encoded'))

class IntraFusion(nn.Module):
    def __init__(self, dimension=128, layers=2, heads=8, dropout=.2,pf_ratio=4):
        super().__init__()
        layer=nn.TransformerEncoderLayer(dimension,heads,dimension*pf_ratio,dropout,
                                         activation='relu',batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
    def forward(self,x):return self.encoder(x)

class TriModalFusion(nn.Module):
    def __init__(self, dimension=128, layers=2, heads=8, dropout=.2,pf_ratio=4):
        super().__init__()
        def layer():
            return nn.ModuleList([
                nn.MultiheadAttention(dimension,heads,dropout=dropout,batch_first=True),
                nn.LayerNorm(dimension),
                nn.Sequential(nn.Linear(dimension,pf_ratio*dimension),nn.ReLU(),
                              nn.Dropout(dropout),nn.Linear(pf_ratio*dimension,dimension)),
                nn.LayerNorm(dimension),nn.Dropout(dropout)])
        self.layers=nn.ModuleList([layer() for _ in range(layers)])
    def forward(self,compound,protein1,protein2):
        x=torch.stack([compound,protein1,protein2],dim=1)
        # A query sees the other two entities. Molecule Q attends jointly
        # to the protein1/protein2 K,V specified in manuscript Eq. 2.
        mask=torch.eye(3,dtype=torch.bool,device=x.device)
        for attention,norm1,ffn,norm2,dropout in self.layers:
            context,_=attention(x,x,x,attn_mask=mask,need_weights=False)
            x=norm1(x+dropout(context))
            x=norm2(x+dropout(ffn(x)))
        return tuple(x.unbind(dim=1))

SETTINGS=json.loads(Path(__file__).with_name('config.json').read_text(encoding='utf-8'))
MODEL_ID='TriGlue_manuscript_reference_v2'

class TriComplexClassifier(nn.Module):
    MOLFORMER='ibm-research/MoLFormer-XL-both-10pct'
    def __init__(self,prot1d_dim=21,dropout=.2,enable_smiles_cache=False):
        super().__init__()
        d=SETTINGS['latent_dim']
        self.tokenizer=AutoTokenizer.from_pretrained(self.MOLFORMER,trust_remote_code=True,local_files_only=True)
        self.molf=AutoModel.from_pretrained(self.MOLFORMER,trust_remote_code=True,local_files_only=True)
        self.molf.requires_grad_(False)
        self.seq_proj=nn.Linear(self.molf.config.hidden_size,d)
        self.prot_textcnn=TextCNN(prot1d_dim,d,SETTINGS['textcnn_filters'],SETTINGS['textcnn_kernels'],dropout)
        self.cmp_gnn=GINConvNet(44,d,SETTINGS['gin_hidden_dim'],dropout,SETTINGS['gin_layers'])
        self.prot_gnn=GINConvNet(41,d,SETTINGS['gin_hidden_dim'],dropout,SETTINGS['gin_layers'])
        self.cmp_3d_lin=nn.Sequential(nn.Linear(768,d),nn.BatchNorm1d(d))
        self.prot_3d_lin=nn.Sequential(nn.Linear(1280,d),nn.BatchNorm1d(d))
        self.intra=IntraFusion(d,SETTINGS['attention_layers'],SETTINGS['attention_heads'],dropout,SETTINGS['feedforward_ratio'])
        self.modality_weights=nn.Parameter(torch.ones(3))
        self.cross_fuse=TriModalFusion(d,SETTINGS['attention_layers'],SETTINGS['attention_heads'],dropout,SETTINGS['feedforward_ratio'])
        self.entity_norm=nn.LayerNorm(d)
        hidden=SETTINGS['classifier_hidden_dim']
        self.classifier=nn.Sequential(nn.Linear(6*d+3,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,1))
        self.enable_smiles_cache=enable_smiles_cache
        self.smiles_cache={}
        self.smiles_mask_rate=SETTINGS['smiles_mask_rate']

    @staticmethod
    def mean_residues(x):
        mask=x.abs().sum(-1).ne(0)
        return (x*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)

    def _encode_smiles_cls(self,smiles):
        smiles=list(smiles)
        if not self.training and self.enable_smiles_cache and all(s in self.smiles_cache for s in smiles):
            return torch.stack([self.smiles_cache[s] for s in smiles]).to(self.seq_proj.weight.device)
        tokens=self.tokenizer(smiles,return_tensors='pt',padding=True,truncation=True)
        tokens={k:v.to(self.seq_proj.weight.device) for k,v in tokens.items()}
        if self.training and self.smiles_mask_rate:
            ids=tokens['input_ids'].clone()
            eligible=tokens['attention_mask'].bool()
            for token in self.tokenizer.all_special_ids:eligible &= ids.ne(token)
            selected=(torch.rand_like(ids,dtype=torch.float)<self.smiles_mask_rate)&eligible
            action=torch.rand_like(ids,dtype=torch.float)
            ids[selected & (action<.8)]=self.tokenizer.mask_token_id
            random=selected & (action>=.8) & (action<.9)
            ordinary=torch.tensor([i for i in range(len(self.tokenizer))
                if i not in self.tokenizer.all_special_ids],device=ids.device)
            ids[random]=ordinary[torch.randint(len(ordinary),(int(random.sum()),),device=ids.device)]
            tokens['input_ids']=ids
        self.molf.eval()
        with torch.no_grad():embedding=self.molf(**tokens,return_dict=True).last_hidden_state[:,0,:].detach()
        if not self.training and self.enable_smiles_cache:
            for s,value in zip(smiles,embedding):self.smiles_cache[s]=value.cpu()
        return embedding

    def fuse_modalities(self,representations):
        refined=self.intra(torch.stack(representations,dim=1))
        return (refined*self.modality_weights.softmax(0)[None,:,None]).sum(1)

    def contrastive_loss(self,compound,p1,p2,positive):
        """Eq. 3: molecule/protein positive pairs, excluding each anchor itself."""
        n=len(compound)
        anchors=torch.arange(n,device=compound.device)
        if positive is not None:anchors=anchors[positive.reshape(-1).bool()]
        if not len(anchors):return compound.sum()*0
        z=F.normalize(torch.cat([compound,p1,p2],0),dim=-1)
        similarities=z@z.T/SETTINGS['contrastive_temperature']
        similarities=similarities.masked_fill(torch.eye(3*n,dtype=torch.bool,device=z.device),-torch.inf)
        log_prob=F.log_softmax(similarities,dim=-1)
        return -(log_prob[anchors,anchors+n].sum()+log_prob[anchors,anchors+2*n].sum())

    def forward(self,smiles,c_graph,c_3d,p1_onehot,p1_graph,p1_3d,
                p2_onehot,p2_graph,p2_3d,return_contrastive=False,
                return_features=False,contrastive_mask=None):
        compound=self.fuse_modalities([self.seq_proj(self._encode_smiles_cls(smiles)),
            self.cmp_gnn(c_graph),self.cmp_3d_lin(c_3d.reshape(len(smiles),-1,768).mean(1))])
        p1=self.fuse_modalities([self.prot_textcnn(p1_onehot),self.prot_gnn(p1_graph),
                               self.prot_3d_lin(self.mean_residues(p1_3d))])
        p2=self.fuse_modalities([self.prot_textcnn(p2_onehot),self.prot_gnn(p2_graph),
                               self.prot_3d_lin(self.mean_residues(p2_3d))])
        compound,p1,p2=(self.entity_norm(x) for x in self.cross_fuse(compound,p1,p2))
        similarity=torch.stack([(compound*p1).sum(-1),(compound*p2).sum(-1),(p1*p2).sum(-1)],dim=-1)
        features=torch.cat([similarity,compound,p1,p2,(compound-p1).abs(),
                            (compound-p2).abs(),(p1-p2).abs()],dim=-1)
        logits=self.classifier(features).squeeze(-1)
        result=(logits,self.contrastive_loss(compound,p1,p2,contrastive_mask)) if return_contrastive else (logits,)
        if return_features:result+=(features,)
        return result[0] if len(result)==1 else result
