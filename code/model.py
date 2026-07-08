import os
import pickle
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split
import torch.optim as optim
from TryAttentionBlock import *
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm, global_mean_pool as gep
from torch_geometric.data import Batch


# 加载处理好的蛋白质功能特征
def load_functional_features(dataset):
    """加载蛋白质的功能特征（IPR特征）"""
    print("加载蛋白质功能特征...")
    
    functional_feat_path = f"/home/zouyanling/Direction/MOE_graph_seq/data/dta/{dataset}/features/uniprot_id_to_vector.json"
   
    
    if not os.path.exists(functional_feat_path):
        raise FileNotFoundError(f" 功能特征文件不存在: {functional_feat_path}")
    
    try:
        with open(functional_feat_path, "r") as f:
            functional_data = json.load(f)
        print(f"功能特征文件加载成功: {functional_feat_path}")
        print(f"  包含 {len(functional_data)} 个蛋白质特征")
        
        functional_features = {}
        missing_vector_count = 0
        
        for uniprot_id, vector in functional_data.items():
            uniprot_id = str(uniprot_id)
            
            if vector is None or len(vector) == 0:
                missing_vector_count += 1
                continue
            
            try:
                vector_array = np.array(vector, dtype=np.float32)
                functional_features[uniprot_id] = vector_array
            except Exception as e:
                print(f" 解析向量失败 {uniprot_id}: {e}")
                missing_vector_count += 1
        
        print(f" 成功加载功能特征: {len(functional_features)} 个蛋白质")
        if missing_vector_count > 0:
            print(f" 缺失向量: {missing_vector_count} 个蛋白质")
        
        return functional_features
    
    except Exception as e:
        raise RuntimeError(f" 加载功能特征失败: {e}")


class Encoder(nn.Module):
    def __init__(self, max_len, input_dim, device, hidden_dim=128):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = 7
        self.do = nn.Dropout(0.1)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.fc = nn.Linear(self.input_dim, self.hidden_dim)
        self.ln = nn.LayerNorm(self.hidden_dim)
        self.convs = nn.ModuleList([nn.Conv1d(self.hidden_dim, self.hidden_dim*2, self.kernel_size, padding=(self.kernel_size-1)//2),
                                    nn.Conv1d(self.hidden_dim, self.hidden_dim*2, self.kernel_size, padding=(self.kernel_size-1)//2),
                                    nn.Conv1d(self.hidden_dim, self.hidden_dim*2, self.kernel_size, padding=(self.kernel_size-1)//2)])
        self.max_pool = nn.MaxPool1d(max_len)

    def forward(self, feat_map):
        h_map = self.fc(feat_map)
        h_map = h_map.permute(0,2,1)  
              
        for i, conv in enumerate(self.convs):
            conved = conv(self.do(h_map))
            conved = F.glu(conved, dim=1)
            conved = (conved+h_map)* self.scale
            h_map = conved
        
        pool_map = self.max_pool(h_map).squeeze(-1)  # b,d
        h_map = h_map.permute(0,2,1)
        h_map = self.ln(h_map)    # b, len, d
        return h_map, pool_map



class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=8, k=4,noise_std=1.0):
        """
        Mixture of Experts层
        Args:
            input_dim: 输入特征维度
            output_dim: 输出特征维度
            num_experts: 专家数量
            k: 每个样本选择的专家数量
        """
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.k = k
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.noise_std = noise_std 
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim,512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, output_dim)
            ) for _ in range(num_experts)
        ])
        
        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts),
            # nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 1. 原始 logits
        logits = self.gate(x)                     # [batch, num_experts]

        # 2. 添加噪声（仅训练时添加，推理时不加）
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            noisy_logits = logits + noise
        else:
            noisy_logits = logits
        
        # 3. 用 noisy_logits 计算门控权重（用于专家加权）
        gate_weights = F.softmax(noisy_logits, dim=-1)
        # 选择top-k专家
        topk_weights, topk_indices = torch.topk(gate_weights, self.k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(-1).expand(-1, self.k)
        selected_outputs = expert_outputs[batch_indices, topk_indices]  # [batch, k, output_dim]
    
        # 加权求和
        output = (selected_outputs * topk_weights.unsqueeze(-1)).sum(dim=1)
        

        # 6. 重要性 imp_i = sum(gate_weights) over batch
        imp = gate_weights.sum(dim=0)                # [num_experts]
        mean_imp = imp.mean()
        std_imp = imp.std()
        L_imp = (std_imp / (mean_imp + 1e-8)) ** 2   # 避免除零

        # 7. 负载 load_i = sum(p_i) over batch
        if self.training and self.noise_std > 0:
            # 根据 noisy_logits 计算 top-k 阈值 eta_K
            topk_noisy_vals, _ = torch.topk(noisy_logits, self.k, dim=-1)
            eta_K = topk_noisy_vals[:, -1].unsqueeze(-1)          # [batch, 1]
            t = (eta_K - logits) / self.noise_std                 # [batch, num_experts]
            p_i = 0.5 * (1 - torch.erf(t / np.sqrt(2)))           # [batch, num_experts]
        else:
            p_i = gate_weights

        load = p_i.sum(dim=0)                          # [num_experts]
        mean_load = load.mean()
        std_load = load.std()
        L_load = (std_load / (mean_load + 1e-8)) ** 2
        # print(f"L_imp: {L_imp:.4f}, L_load: {L_load:.4f}")

        # 总均衡损失
        L_blc = 0.5 * (L_imp + L_load)

        return output, L_blc   # 返回总损失及各分量，方便调试
    


#moe_transformer
# ==================== 1. 专家定义：单头 Transformer ====================
class SingleHeadTransformerExpert(nn.Module):
    """单头 Transformer 专家，输入序列，输出药物全局特征和蛋白质全局特征"""
    def __init__(self, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=1,                      # 单头注意力
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, seq, drug_seq_len):
        """
        seq: (batch, seq_len, hidden_dim)
        drug_seq_len: 当前 batch 中药物的矩阵序列长度（用于定位蛋白质全局特征的位置）
        returns:
            drug_out: (batch, hidden_dim)
            prot_out: (batch, hidden_dim)
        """
        out = self.transformer(seq)                       # (batch, seq_len, hidden_dim)
        drug_out = out[:, 0, :]                           # 药物全局 token 在位置 0
        prot_out = out[:, 1 + drug_seq_len, :]            # 蛋白质全局 token 位置
        return drug_out, prot_out


# ==================== 2. MoE 层（每个专家为单头 Transformer）====================
class MoETransformer(nn.Module):
    def __init__(self, drug_global_dim, prot_global_dim, hidden_dim,
                 num_layers=1, num_experts=6, k=4, dropout=0.1, noise_std=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std

        # 门控网络：输入 = 药物全局特征 + 融合后的蛋白质全局特征
        self.gate = nn.Sequential(
            nn.Linear(drug_global_dim + prot_global_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts)
        )

        # 专家列表：每个专家都是一个单头 Transformer
        self.experts = nn.ModuleList([
            SingleHeadTransformerExpert(hidden_dim, num_layers, dropout)
            for _ in range(num_experts)
        ])

    def forward(self, seq, drug_feat, target_fused, drug_seq_len):
        """
        seq: (batch, seq_len, hidden_dim)  输入序列
        drug_feat: (batch, drug_global_dim)  原始药物全局特征
        target_fused: (batch, prot_global_dim)  融合功能特征后的蛋白质全局特征
        drug_seq_len: int, 当前 batch 中药物的矩阵序列长度（用于定位）
        """
        batch_size = seq.size(0)
        device = seq.device

        # ---------- 门控计算 ----------
        gate_input = torch.cat([drug_feat, target_fused], dim=1)   # (batch, drug_global_dim+prot_global_dim)
        logits = self.gate(gate_input)                             # (batch, num_experts)

        # 训练时添加噪声
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            noisy_logits = logits + noise
        else:
            noisy_logits = logits

        gate_weights = F.softmax(noisy_logits, dim=-1)            # (batch, num_experts)

        # 选择 top-k 专家
        topk_weights, topk_indices = torch.topk(gate_weights, self.k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # ---------- 负载均衡损失计算（参考 MoE 类）----------
        # 重要性：impor_i = sum(gate_weights) over batch
        imp = gate_weights.sum(dim=0)                             # (num_experts,)
        mean_imp = imp.mean()
        std_imp = imp.std()
        L_imp = (std_imp / (mean_imp + 1e-8)) ** 2

        # 负载：根据 noisy_logits 计算每个专家的保留概率 p_i
        if self.training and self.noise_std > 0:
            topk_noisy_vals, _ = torch.topk(noisy_logits, self.k, dim=-1)
            eta_K = topk_noisy_vals[:, -1].unsqueeze(-1)          # (batch, 1)
            t = (eta_K - logits) / self.noise_std                 # (batch, num_experts)
            p_i = 0.5 * (1 - torch.erf(t / np.sqrt(2)))           # (batch, num_experts)
        else:
            p_i = gate_weights
        load = p_i.sum(dim=0)                                     # (num_experts,)
        mean_load = load.mean()
        std_load = load.std()
        L_load = (std_load / (mean_load + 1e-8)) ** 2
        L_blc = 0.5 * (L_imp + L_load)

        # ---------- 加权融合各专家的输出 ----------
        drug_out_total = torch.zeros(batch_size, self.hidden_dim, device=device)
        prot_out_total = torch.zeros(batch_size, self.hidden_dim, device=device)

        # 按专家分组：每个专家处理分配到的样本
        # 构建一个字典：expert_idx -> list of (sample_idx, weight)
        expert_to_samples = {i: [] for i in range(self.num_experts)}
        for b in range(batch_size):
            for idx_in_topk in range(self.k):
                expert_idx = topk_indices[b, idx_in_topk].item()
                weight = topk_weights[b, idx_in_topk].item()
                expert_to_samples[expert_idx].append((b, weight))

        # 对每个专家，批量处理其对应的样本
        for expert_idx, samples in expert_to_samples.items():
            if not samples:
                continue
            indices = [s[0] for s in samples]
            weights = torch.tensor([s[1] for s in samples], device=device).view(-1, 1)   # (m,1)

            # 提取这些样本的序列
            batch_seq = seq[indices]                            # (m, seq_len, hidden_dim)
            # 专家前向，得到 drug 和 prot 输出
            drug_sub, prot_sub = self.experts[expert_idx](batch_seq, drug_seq_len)   # 均为 (m, hidden_dim)
            # 加权累加
            drug_out_total[indices] += drug_sub * weights
            prot_out_total[indices] += prot_sub * weights

        return drug_out_total, prot_out_total, L_blc



class GatedFusion(nn.Module):
    def __init__(self, prot_dim, func_dim, use_relu=True, dropout=0.1):
        super().__init__()
        self.func_proj = nn.Linear(func_dim, prot_dim)
        # 门控生成网络
        self.gate_net = nn.Sequential(
            nn.Linear(prot_dim * 2, prot_dim),
            nn.Sigmoid()  # 输出门控值在 0~1 之间
        )
        self.use_relu = use_relu
        if use_relu:
            self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, prot_feat, func_feat):
        func_proj = self.func_proj(func_feat)  # [batch, prot_dim]
        gate_input = torch.cat([prot_feat, func_proj], dim=1)
        gate = self.gate_net(gate_input)  # [batch, prot_dim]，每个元素 ∈ (0,1)
        fused = gate * prot_feat + (1 - gate) * func_proj
        if self.use_relu:
            fused = self.relu(fused)
        fused = self.dropout(fused)
        return fused


class CasMoeDTA(nn.Module):
    def __init__(self, drug_global_dim,      # 药物全局特征维度（如 Mol2Vec 300）     
                 prot_global_dim,      # 蛋白质全局特征维度（如 ProtTrans 池化后 1024）
                 func_dim,     # 蛋白质功能特征维度（如 IPR 512）
                 hidden_dim=256,
                 num_heads=4,
                 num_layers=1,
                 dropout=0.1,
                 moe_output_dim=256,          # MoE 输出维度
                 drug_max_len=50,     # 药物矩阵序列最大长度
                 prot_max_len=1280,    # 蛋白质矩阵序列最大长度
                 device='cuda:1'
                 ):
        super(CasMoeDTA, self).__init__()

        self.dropout = nn.Dropout(0.1)  
        #CPI分支预测
        self.fun_feat_combine = GatedFusion(prot_dim=prot_global_dim, func_dim=func_dim, use_relu=True, dropout=dropout)# 输出维度是prot_global_dim
        # 四个部分的投影层，统一到 hidden_dim
        self.drug_global_proj = nn.Sequential(
            nn.Linear(drug_global_dim, drug_global_dim),
            nn.PReLU(),
            nn.Linear(drug_global_dim, hidden_dim),
            nn.PReLU()
        )
        # 药物矩阵编码器：输入维度 drug_global_dim（如 300），输出维度 hidden_dim
        self.drug_matrix_encoder = Encoder(
            max_len=drug_max_len,
            input_dim=drug_global_dim,
            device=device,
            hidden_dim=hidden_dim
        )
        self.prot_global_proj = nn.Sequential(
            nn.Linear(prot_global_dim, prot_global_dim),
            nn.PReLU(),
            nn.Linear(prot_global_dim, hidden_dim),
            nn.PReLU()
        )
        # 蛋白质矩阵编码器：输入维度 prot_global_dim（如 1024），输出维度 hidden_dim
        self.prot_matrix_encoder = Encoder(
            max_len=prot_max_len,
            input_dim=prot_global_dim,
            device=device,
            hidden_dim=hidden_dim
        )
    
        self.moe_transformer = MoETransformer(
            drug_global_dim=drug_global_dim,
            prot_global_dim=prot_global_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_experts=6,
            k=4,
            dropout=dropout,
            noise_std=1.0
        )

        self.drug_target_moe = MoE(input_dim=hidden_dim*2, output_dim=256, num_experts=6, k=4)
        self.mlp_pred1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )  

       
        #最终预测
        self.mlp_pred=nn.Linear(2, 1)
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # def forward(self, drug_feat, drug_matrix,target_feat, target_matrix,func_feat, data_drug, data_prot):
    def forward(self, drug_feat, drug_matrix,target_feat, target_matrix,func_feat):   
        target_fused= self.fun_feat_combine(target_feat, func_feat)
        # 1. 投影到统一维度
        drug_global_h = self.drug_global_proj(drug_feat)                # (B, H)
        drug_matrix_h, drug_global_h_encoder = self.drug_matrix_encoder(drug_matrix)                # (B, Ld, H)
        prot_global_h = self.prot_global_proj(target_fused)                # (B, H)
        prot_matrix_h, prot_global_h_encoder = self.prot_matrix_encoder(target_matrix)                # (B, Lp, H)

        # 2. 构建统一序列：药物全局 + 药物矩阵 + 蛋白质全局 + 蛋白质矩阵
        drug_global_h = drug_global_h.unsqueeze(1)                         # (B, 1, H)
        prot_global_h = prot_global_h.unsqueeze(1)                         # (B, 1, H)
        seq = torch.cat([drug_global_h, drug_matrix_h, prot_global_h, prot_matrix_h], dim=1) # seq 形状: (B, 1 + Ld + 1 + Lp, H)
        
        # 4. 通过 MoETransformer 得到药物和蛋白质全局表征
        drug_seq_len = drug_matrix_h.size(1)   # 当前 batch 的药物矩阵序列长度（已 padding）
        drug_out, prot_out, moe_trans_loss = self.moe_transformer(seq, drug_feat, target_fused, drug_seq_len)

        drug_target_feat = torch.cat([drug_out, prot_out], dim=1)       # (B, H*2)
        drug_target_moe_out, moe_loss = self.drug_target_moe(drug_target_feat)

        # 6.分支1预测
        pred1 = self.mlp_pred1(drug_target_moe_out).squeeze(-1)  # (B,)

        # 总损失 = 两个 MoE 的负载平衡损失之和（可根据需要加权）
        # total_moe_loss = moe_trans_loss + moe_loss
        total_moe_loss = 0.0

        return pred1, total_moe_loss
    

