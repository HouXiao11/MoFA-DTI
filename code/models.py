import torch.nn as nn
import torch.nn.functional as F
import torch
from dgllife.model.gnn import GCN
from Attention import MyAttention
from einops import repeat

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss

def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss

def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


class MoFADTI(nn.Module):
    def __init__(self, device='cuda', **config):
        super(MoFADTI, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        protein_num_head = config['PROTEIN']['NUM_HEAD']
        cross_num_head = config['CROSSINTENTION']['NUM_HEAD']
        cross_emb_dim = config['CROSSINTENTION']['EMBEDDING_DIM']
        cross_layer = config['CROSSINTENTION']['LAYER']

        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)
        
        self.protein_extractor = ProteinBiGRUWithPositionEncodingCNN(protein_emb_dim,hidden_dim=256,output_dim=128,num_layers=1,dropout=0.5)
        
        self.cross_intention = MyAttention(embed_dim=cross_emb_dim, trans_layers=6, num_heads=cross_num_head, dropout=0.1, 
                            kernel_size=3, num_filters=128,device=device)

        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, bg_d, v_p, mode="train"):

        v_d = self.drug_extractor(bg_d)
        v_p = self.protein_extractor(v_p)

        f, v_d, v_p, att = self.cross_intention(drug=v_d, protein=v_p)

        score = self.mlp_classifier(f)
        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, att

class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):#x.shpae[64, 256]
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

class ProteinBiGRUWithPositionEncodingCNN(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,output_dim,num_layers=1, dropout=0.5,padding=True):
        super(ProteinBiGRUWithPositionEncodingCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)

        # position_encoding
        self.position_encoding = self._generate_position_encoding(max_seq_len=1200,embedding_dim=128)

        # BiGRU for feature extraction
        self.bigru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # CNN layer for feature extraction
        self.conv1 = nn.Conv1d(in_channels=hidden_dim*2, out_channels=hidden_dim, kernel_size=3, padding=1)  # Example kernel size
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def _generate_position_encoding(self, max_seq_len, embedding_dim):
        """生成正弦位置编码"""
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embedding_dim))
        position_encoding = torch.zeros(max_seq_len, embedding_dim)
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        return position_encoding

    def forward(self, v): # (64,1200)

        # batch_size, seq_len = v.size()
        v = self.embedding(v.long()) # (64,1200,128)

        # 添加位置编码
        seq_len = v.size(1)
        pos_encoding = self.position_encoding[:seq_len, :].unsqueeze(0).to(v.device)
        v = v + pos_encoding  # [batch_size, seq_len, embedding_dim]

        v, _ = self.bigru(v)

        # CNN feature extraction
        v = v.permute(0, 2, 1)  # Shape: (batch_size, 256, seq_len)
        v = self.conv1(v)  # Apply first CNN layer
        v = torch.relu(v)
        v = self.conv2(v)  # Apply second CNN layer
        v = torch.relu(v)
        # print(v.shape) # [64, 256, 1200]
        v = self.pool(v)  # Max pooling
        v = v.permute(0, 2, 1)  # Shape: (batch_size, reduced_seq_len, feature_dim)
        
        v = self.fc(v)

        return v
