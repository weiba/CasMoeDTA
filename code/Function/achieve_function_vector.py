import pandas as pd
import numpy as np
import json

# 读取现有矩阵
matrix_df = pd.read_csv("protein_ipr_matrix.csv")

print(f"矩阵形状: {matrix_df.shape}")
print(f"前5个蛋白: {matrix_df['uniprot_id'].head().tolist()}")

# 创建ID到向量的映射
id_to_vector = {}
feature_columns = [col for col in matrix_df.columns if col != 'uniprot_id']

for idx, row in matrix_df.iterrows():
    uniprot_id = row['uniprot_id']
    vector = row[feature_columns].astype(int).values.tolist()
    id_to_vector[uniprot_id] = vector

# 保存为JSON格式
with open("uniprot_id_to_vector.json", "w") as f:
    json.dump(id_to_vector, f)

print(f"\n已创建ID到向量的映射")
print(f"总蛋白数: {len(id_to_vector)}")
print(f"向量维度: {len(feature_columns)}")
print(f"文件已保存: uniprot_id_to_vector.json")

# 示例使用
sample_id = matrix_df['uniprot_id'].iloc[0]
print(f"\n示例 - 蛋白 {sample_id}:")
print(f"向量: {id_to_vector[sample_id][:10]}...")  # 显示前10个值
print(f"向量长度: {len(id_to_vector[sample_id])}")
print(f"非零元素数: {sum(id_to_vector[sample_id])}")

# 也可以保存为CSV格式（每行一个ID和向量）
with open("uniprot_vectors.csv", "w") as f:
    f.write("uniprot_id,vector\n")
    for uniprot_id, vector in id_to_vector.items():
        vector_str = ','.join(map(str, vector))
        f.write(f"{uniprot_id},{vector_str}\n")

print(f"\nCSV格式已保存: uniprot_vectors.csv")