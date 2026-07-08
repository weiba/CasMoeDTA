
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
import time 
import pickle
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.optim as optim
from TryAttentionBlock import *
from torch_geometric.data import Batch
from utils import load_data, rmse, mse, pearson, spearman, ci, roc_auc, pr_auc, rm2
from model import CasMoeDTA , load_functional_features



#使用混合精度训练
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")


# #加载预计算的药物特征
def load_compound_required_features(dataset):
    """加载药物特征"""
    print("=" * 60)
    print("开始加载药物特征...")
    
    dataset_path = f"/home/zouyanling/Direction/MOE_graph_seq/data/dta/{dataset}/features"
    drug_path = os.path.join(dataset_path, "compound_Mol2Vec300.pkl")
    drug_matrix_path = os.path.join(dataset_path, "compound_Atom2Vec300.pkl")
    
    with open(drug_path, "rb") as f:
        drug_features = pickle.load(f)
    print(f"药物特征加载完成: {len(drug_features)} 个药物")

    with open(drug_matrix_path, "rb") as f:
        drug_matrix_features = pickle.load(f)
    print(f"药物矩阵特征加载完成: {len(drug_matrix_features)} 个药物")
    
    features = {}
    for pid, drug_feat in drug_features.items():
        pid_str = str(pid)
        features[pid_str] = drug_feat

    features_matrix = {}
    # counts = 0
    for pid, drug_matrix_feat in drug_matrix_features.items():
        pid_str = str(pid)
        features_matrix[pid_str] = drug_matrix_feat
    print(f"药物特征维度: {drug_feat.shape}, 药物矩阵特征维度: {drug_matrix_feat.shape}")
    print("药物特征加载完成")
    print("=" * 60)
    return features, features_matrix

# 加载预计算的蛋白质特征
def load_precomputed_features(dataset):
    """
    加载预计算的蛋白质特征
    Args:
        dataset: 数据集名称
        feature_type: 特征类型，如 "ProtTransBertBFD" 或 "aas_ProtTransBertBFD1024"
    Returns:
        prot_features_dict: 蛋白质ID到特征的映射
    """

    path = f"/home/zouyanling/Direction/MOE_graph_seq/data/dta/{dataset}/features/aas_ProtTransBertBFD1024_updated.pkl"
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    # 提取蛋白质级向量（已池化的特征）
    prot_features_vector = {}
    prot_features_matrix = {}
    count = 0 
    for pid, (vector, matrix) in data.items():
        # vector: 蛋白质级表示 [1024,]
        # matrix: 氨基酸级表示 [seq_len, 1024]
        prot_features_vector[pid] = vector  # 使用蛋白质级表示
        prot_features_matrix[pid] = matrix

        

    print(f"蛋白质级向量维度: {vector.shape}, 氨基酸级向量维度: {matrix.shape}")
    return prot_features_vector, prot_features_matrix

#自定义数据集类
class DTIDataset(torch.utils.data.Dataset):
    def __init__(self, df, comp_feat,comp_matrix_feat, prot_feat,prot_matrix_feat, prot_func_feat):
        self.df = df
        self.df = df
        self.comp_feat = comp_feat
        self.comp_matrix_feat = comp_matrix_feat
        self.prot_feat = prot_feat
        self.prot_matrix_feat = prot_matrix_feat
        self.prot_func_feat = prot_func_feat     # 字典或 None


       
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cid = str(row['cid'])
        pid = str(row['pid'])
        label = row['label']

        # 药物特征向量（基础特征 + 指纹）
        drug_vec = self.comp_feat[cid]
        drug_matrix_vec = self.comp_matrix_feat.get(cid)

        # 蛋白质特征向量（基础特征 + 功能特征）
        prot_vec = self.prot_feat[pid]
        if self.prot_func_feat is not None:
            func_vec = self.prot_func_feat.get(pid)
        prot_matrix_vec = self.prot_matrix_feat.get(pid)
        # 图对象
        return {
            'drug_vec': torch.tensor(drug_vec, dtype=torch.float),
            'drug_matrix_vec': torch.tensor(drug_matrix_vec, dtype=torch.float),
            'prot_vec': torch.tensor(prot_vec, dtype=torch.float),
            'prot_matrix_vec': torch.tensor(prot_matrix_vec, dtype=torch.float),
            'func_vec': torch.tensor(func_vec, dtype=torch.float) if func_vec is not None else None,
            'label': torch.tensor(label, dtype=torch.float)
        }

#合并数据集形成batch
def collate_fn(batch):
    drug_max_len = 50  # 药物分子数最大长度（根据数据集统计调整）
    prot_max_len = 1280
    drug_vecs = torch.stack([item['drug_vec'] for item in batch])
    #处理药物特征矩阵，将列表转换为张量
    drug_matrix_list = [item['drug_matrix_vec'] for item in batch] 
    drug_feat_dim = drug_matrix_list[0].size(1)    
    drug_matrix_padded = torch.zeros(len(batch), drug_max_len, drug_feat_dim, dtype=drug_matrix_list[0].dtype)
    for i, mat in enumerate(drug_matrix_list):
        length = min(mat.size(0), drug_max_len)                                 # 取实际长度和最大长度的较小值
        drug_matrix_padded[i, :length] = mat[:length] 
    prot_vecs = torch.stack([item['prot_vec'] for item in batch])
    # prot_matrix_vecs = torch.stack([item['prot_matrix_vec'] for item in batch])
    prot_matrix_list = [item['prot_matrix_vec'] for item in batch]
    prot_feat_dim = prot_matrix_list[0].size(1)
    prot_matrix_padded = torch.zeros(len(batch), prot_max_len, prot_feat_dim, dtype=prot_matrix_list[0].dtype)
    for i, mat in enumerate(prot_matrix_list):
        length = min(mat.size(0), prot_max_len)
        prot_matrix_padded[i, :length] = mat[:length]
    func_vecs = torch.stack([item['func_vec'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    return drug_vecs, drug_matrix_padded,prot_vecs, prot_matrix_padded, func_vecs,  labels

# ==================== 训练函数 ====================
def train_model(model, train_loader, criterion, optimizer, eval_metric=None, num_epochs=100, patience=50, min_delta=0.01,dataset_name="", setting="",fold=1):
    model.train()
    best_train_loss = float('inf')  
    patience_counter = 0
    accumulation_steps = 2

    # 动态生成模型保存路径
    
    model_save_path = f'best_{dataset_name}_{setting}_model_fold{fold}_GRU.pth'
    print(f"设置的patience: {patience}, 模型保存路径: {model_save_path}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        total_moe_loss = 0.0
        optimizer.zero_grad() 

        for step, (drug_vecs, drug_matrix_padded, prot_vecs, prot_matrix_padded, func_vec, labels) in enumerate(train_loader):   
            drug_vecs = drug_vecs.to(device)
            drug_matrix_padded = drug_matrix_padded.to(device)
            prot_vecs = prot_vecs.to(device)
            prot_matrix_padded = prot_matrix_padded.to(device)
            func_vec = func_vec.to(device)
            labels = labels.to(device)

            with autocast():
                predictions,moe_loss = model(drug_vecs, drug_matrix_padded, prot_vecs, prot_matrix_padded, func_vec)
                pre_loss = criterion(predictions, labels)
                loss = pre_loss + 0.01 * moe_loss
                # 缩放损失，以便在累积梯度时保持数值稳定
                scaled_loss = loss / accumulation_steps  # 平均损失，使累积梯度的规模与单次更新一致
            
            scaler.scale(scaled_loss).backward()  # 反向传播累积梯度
            
            total_loss += loss.item()  # 累加原始损失用于日志
            total_moe_loss += moe_loss.item() 

            # 每 accumulation_steps 步执行一次优化器更新
            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()  # 清零梯度，准备下一组累积

        avg_train_loss = total_loss / len(train_loader)
        avg_moe_loss = total_moe_loss / len(train_loader)

        # 打印训练信息（每10个epoch或第一个epoch）
        if (epoch + 1) % 10 == 0 or epoch ==1 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}')

        # 早停检查：基于训练损失
        improvement = best_train_loss - avg_train_loss
        if improvement > min_delta:
            best_train_loss = avg_train_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)   # 保存最佳模型
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'早停于 epoch {epoch+1}，连续 {patience} 个epoch损失下降未超过 {min_delta}')
                break

    # 加载最佳模型并清理临时文件
    model.load_state_dict(torch.load(model_save_path))

    torch.cuda.empty_cache()


def kfold_validation(task: str, dataset: str, setting: str, preset=None, ex_model=[]) -> None:
    """
    Perform k-fold validation for the given task, dataset, and setting.
    """
    assert task in ["dti", "dta", "moa"], "task should be in ('dti', 'dta', 'moa')."
    if task == "dta":
        assert dataset in [
            "davis",
            "kiba",
        ], f"dataset should be in ('davis','kiba') for {task} task."
        dataset_path = "../data/dta/" + dataset + "/"
        k_folds = 4
        eval_metric = None
        res_all = pd.DataFrame(columns=["RMSE", "MSE", "Pearson", "Spearman", "CI", "rm2"])

    else:
        print(f"当前任务: {task}")

    assert (
        setting in ["warm_start", "drug_coldstart", "protein_coldstart"]
    ), "validation setting should be in ('warm_start', 'drug_coldstart', 'protein_coldstart')."



    prot_feat, prot_matrix_feat = load_precomputed_features(dataset)
    comp_feat, comp_matrix_feat = load_compound_required_features(dataset)
    prot_func_feat = load_functional_features(dataset)
    if prot_func_feat and len(prot_func_feat) > 0:
        sample_func = next(iter(prot_func_feat.values()))
        func_dim = len(sample_func)
        print(f"功能特征维度: {func_dim}")
    else:
        func_dim = 0  # 如果没有功能特征，设为0

    folds_path = dataset_path + "data_folds/" + setting + "/"
    print(f"Evaluating the model on {dataset} dataset under {setting} setting ...")
    
    for i in range(k_folds):
        print("fold:", i + 1)
        i=i+1
        print(f"\n>>> Starting fold {i+1}")
        train_data, test_data = load_data(folds_path, i, comp_feat, prot_feat)
        batch_size = 64

        # 根据索引创建子集 DataFrame
        train_dataset = DTIDataset(train_data, comp_feat, comp_matrix_feat, prot_feat, prot_matrix_feat, prot_func_feat)
        test_dataset = DTIDataset(test_data, comp_feat, comp_matrix_feat, prot_feat, prot_matrix_feat, prot_func_feat)

        # 构建数据集
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
            num_workers=8, pin_memory=True, persistent_workers=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
            num_workers=4, pin_memory=True
        )

        
        print("开始训练MoE模型...")
        model = CasMoeDTA(drug_global_dim=300, prot_global_dim=1024,func_dim=func_dim).to(device)

        criterion = nn.MSELoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
        train_model(
            model, train_loader, criterion, optimizer,
            num_epochs=500, patience=500, min_delta=0.01,
            dataset_name=dataset, setting=setting,fold=i+1
        )
        
        # ==================== 使用训练好的MoE提取特征 ==================== 
        print("直接在测试集上评估训练好的MoE模型...")
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for drug_vecs,drug_matrix_padded,prot_vecs, prot_matrix_padded, func_vec, labels in test_loader: 
                drug_vecs = drug_vecs.to(device)
                drug_matrix_padded = drug_matrix_padded.to(device)
                prot_vecs = prot_vecs.to(device)
                prot_matrix_padded = prot_matrix_padded.to(device)
                func_vec = func_vec.to(device)
                labels = labels.to(device)  # 标签也建议移动，虽然计算 loss 时可能自动转换，但保持统一更好
                predictions, _ = model(drug_vecs, drug_matrix_padded, prot_vecs, prot_matrix_padded, func_vec)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        if task == "dta":
            ret = [rmse(all_labels, all_preds), mse(all_labels, all_preds),
                pearson(all_labels, all_preds), spearman(all_labels, all_preds),
                ci(all_labels, all_preds), rm2(all_labels, all_preds)]
            print(f"fold: {i+1}, RMSE: {ret[0]}, MSE: {ret[1]}, Pearson: {ret[2]}, "
                f"Spearman: {ret[3]}, CI: {ret[4]}, RM2: {ret[5]}")
            res_all.loc[i] = ret
        else:  # dti 或 moa
            print("当前任务: ", task)

    if task == "dta":
        print("\n所有fold结果：")
        print(res_all)
        print("\n平均结果：")
        print(res_all.mean(axis=0))
        os.makedirs("../results/", exist_ok=True)
        res_all.to_csv(f"../results/{dataset}_{setting}.csv", index=None, sep="\t")

if __name__ == "__main__":
    start_time = time.time()
    task, dataset, setting = sys.argv[1], sys.argv[2], sys.argv[3]
    kfold_validation(task, dataset, setting)
    # 计算并打印总运行时间
    end_time = time.time()
    total_minutes = (end_time - start_time) / 60
    print(f"程序总运行时间: {total_minutes:.2f} 分钟")