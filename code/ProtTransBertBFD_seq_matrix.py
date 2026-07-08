import torch
import torch.nn as nn
import torch.nn.functional as F  
import pandas as pd
from tqdm import tqdm
import esm
import numpy as np
import os
import json
import re
import math
import pickle

# 使用预提取的ProtTransBertBFD特征进行MoPE融合
class PrecomputedProtTransBFDMoPE(nn.Module):
    """
    使用预提取的ProtTransBertBFD特征进行MoPE融合
    """
    def __init__(self, num_experts=8, func_dim=526, hidden_dim=1024, 
                 use_residual=True, use_concatenate=False):
        super().__init__()
        
        self.hidden_dim = hidden_dim  # ProtTransBertBFD特征维度
        self.func_dim = func_dim
        self.num_experts = num_experts
        self.use_residual = use_residual
        self.use_concatenate = use_concatenate
        
        # MoPE组件
        self.routing_dim = 256
        
        # 1. 可训练的专家特征库 [num_experts, hidden_dim]
        self.expert_features = nn.Parameter(
            torch.randn(num_experts, hidden_dim)
        )
        nn.init.xavier_normal_(self.expert_features)
        
        # 2. 冻结的路由关键表示 [num_experts, routing_dim]
        self.frozen_keys = nn.Parameter(
            torch.randn(num_experts, self.routing_dim),
            requires_grad=False
        )
        nn.init.orthogonal_(self.frozen_keys)
        
        # 3. 查询投影网络：将ProtTrans特征+功能特征投影到路由空间
        self.query_projector = nn.Sequential(
            nn.Linear(hidden_dim + func_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.routing_dim)
        )
        
        # # 4. 融合层
        # if self.use_concatenate:
        #     self.fusion_layer = nn.Sequential(
        #         nn.Linear(hidden_dim * 2, 1024),
        #         nn.ReLU(),
        #         nn.Dropout(0.1),
        #         nn.Linear(1024, hidden_dim)
        #     )
        
        # 5. 最终映射到1024维（如果需要）
        self.final_projection = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    
    def forward(self, prot_features, func_features):
        """
        前向传播
        Args:
            prot_features: 预计算的ProtTransBertBFD特征 [batch_size, hidden_dim]
            func_features: 功能特征 [batch_size, func_dim]
        Returns:
            final_representation: 最终表示 [batch_size, 1024]
            gate_weights: 门控权重 [batch_size, num_experts]
        """
        batch_size = prot_features.size(0)
        
        # ============ 1. 计算查询向量 ============
        combined_input = torch.cat([prot_features, func_features], dim=1)
        query_vectors = self.query_projector(combined_input)  # [batch_size, routing_dim]
        
        # ============ 2. 计算路由权重 ============
        # 查询与冻结关键表示计算相似度
        moe_logits = torch.mm(query_vectors, self.frozen_keys.t())  # [batch_size, num_experts]
        
        temperature = 0.1
        moe_logits = moe_logits / temperature
        
        if self.training:
            noise = torch.randn_like(moe_logits) * 0.1
            moe_logits = moe_logits + noise
        
        gate_weights = F.softmax(moe_logits, dim=-1)  # [batch_size, num_experts]
        
        # ============ 3. 加权组合专家特征 ============
        dynamic_features = torch.einsum('bk,kd->bd', gate_weights, self.expert_features)
        
        # ============ 4. 与原始特征融合 ============
        if self.use_concatenate:
            combined = torch.cat([prot_features, dynamic_features], dim=1)
            fused_features = self.fusion_layer(combined)
        elif self.use_residual:
            fused_features = prot_features + dynamic_features
        else:
            fused_features = dynamic_features
        
        # ============ 5. 最终投影 ============
        final_representation = self.final_projection(fused_features)  # [batch_size, 1024]
        
        return final_representation, gate_weights