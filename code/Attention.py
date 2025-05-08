import torch
import torch.nn as nn
from einops import reduce

class CrossAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.5):
        super(CrossAttention, self).__init__()
        # 定义用于交叉注意力的线性变换
        self.query_fc = nn.Linear(model_dim, model_dim)
        self.key_fc = nn.Linear(model_dim, model_dim)
        self.value_fc = nn.Linear(model_dim, model_dim)
        
        # Scaled Dot-Product Attention
        self.attn = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, dropout=dropout)
        
    def forward(self, query, key, value):
        # 对查询、键、值进行线性变换
        query = self.query_fc(query)  # (batch_size, seq_len, model_dim)
        key = self.key_fc(key)  # (batch_size, seq_len, model_dim)
        value = self.value_fc(value)  # (batch_size, seq_len, model_dim)
        
        # 转换为 (seq_len, batch_size, model_dim) 的形状
        query = query.permute(1, 0, 2)  # (seq_len, batch_size, model_dim)
        key = key.permute(1, 0, 2)  # (seq_len, batch_size, model_dim)
        value = value.permute(1, 0, 2)  # (seq_len, batch_size, model_dim)
        
        # 交叉注意力计算
        output, _ = self.attn(query, key, value)
        
        return output.permute(1, 0, 2)  # 转回 (batch_size, seq_len, model_dim)

class MyAttention(nn.Module):
    def __init__(self, embed_dim, trans_layers=6, num_heads=8, dropout=0.1, kernel_size=3, 
                 num_filters=128, device='cuda'):
        super(MyAttention, self).__init__()

        self.drug_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout),
            num_layers=trans_layers
        )

        self.protein_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout),
            num_layers=trans_layers
        )

        self.conv_maxpool = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, 
                      out_channels=num_filters, 
                      kernel_size=kernel_size, 
                      padding=kernel_size // 2),  # 保持序列长度
            nn.AdaptiveMaxPool1d(1)  # Global max pooling
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling

        # 交叉注意力模块 output:(batch_size, seq_len, model_dim)
        self.cross_attention = CrossAttention(model_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        
        self.alpha = 0.2

    def forward(self, drug, protein):


        drug = drug.permute(1,0,2) # 需要转置为 (seq_len, batch_size, input_dim) 以适应 Transformer
        trans_drug = self.drug_transformer(drug) 

        pooled_drug = self.avg_pool(trans_drug.permute(1,2,0)).squeeze(-1)
        drug_0 = pooled_drug

        protein = protein.permute(1,0,2) # 需要转置为 (seq_len, batch_size, input_dim) 以适应 Transformer
        trans_protein = self.protein_transformer(protein)

        protein_0 = self.conv_maxpool(trans_protein.permute(1, 2, 0)).squeeze(-1)

        drug_ca = self.cross_attention(trans_drug.permute(1,0,2), trans_protein.permute(1,0,2), trans_protein.permute(1,0,2))  
        protein_ca = self.cross_attention(trans_protein.permute(1,0,2), trans_drug.permute(1,0,2), trans_drug.permute(1,0,2))  

        drug_1 = self.avg_pool(drug_ca.permute(0,2,1)).squeeze(-1)

        protein_1 = self.conv_maxpool(protein_ca.permute(0,2,1)).squeeze(-1)

        f_0 = torch.cat((drug_0, protein_0), dim=1)
        f_1 = torch.cat((drug_1, protein_1), dim=1)

        f = f_0 + self.alpha * f_1

        return f, drug_1, protein_1, None
