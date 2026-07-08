import os
import glob
import pandas as pd
from collections import Counter
import numpy as np
import re

# ==================== 新增: 文件诊断函数 ====================
def diagnose_files(folder_path):
    """
    诊断文件夹中的所有文件，找出问题文件
    """
    print("="*60)
    print("开始诊断文件...")
    print("="*60)
    
    tsv_files = glob.glob(os.path.join(folder_path, "*.tsv"))
    print(f"找到 {len(tsv_files)} 个.tsv文件")
    
    # 统计信息
    file_stats = []
    uniprot_id_counter = Counter()
    problem_files = []
    valid_files = 0
    
    for file_idx, file_path in enumerate(tsv_files):
        file_name = os.path.basename(file_path)
        stats = {
            'file_name': file_name,
            'uniprot_id': None,
            'has_uniprot_id': False,
            'has_ipr': False,
            'ipr_count': 0,
            'line_count': 0,
            'file_size': os.path.getsize(file_path),
            'status': '正常',
            'error': None
        }
        
        try:
            # 检查文件大小
            if stats['file_size'] == 0:
                stats['status'] = '空文件'
                stats['error'] = '文件为空'
                problem_files.append(stats)
                file_stats.append(stats)
                continue
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                stats['line_count'] = len(lines)
                
                if len(lines) == 0:
                    stats['status'] = '空文件'
                    stats['error'] = '文件内容为空'
                    problem_files.append(stats)
                    file_stats.append(stats)
                    continue
                
                # 提取第一行的UniProt ID
                first_line = lines[0].strip()
                if first_line:
                    parts = first_line.split('\t')
                    if len(parts) > 0:
                        uniprot_id = parts[0].strip()
                        if uniprot_id:
                            stats['uniprot_id'] = uniprot_id
                            stats['has_uniprot_id'] = True
                            
                            # 统计UniProt ID
                            uniprot_id_counter[uniprot_id] += 1
                            
                            # 检查是否包含IPR
                            ipr_matches = re.findall(r'IPR\d{6}', ''.join(lines))
                            ipr_count = len(set(ipr_matches))
                            stats['ipr_count'] = ipr_count
                            stats['has_ipr'] = ipr_count > 0
                            
                            valid_files += 1
                        else:
                            stats['status'] = '无UniProt ID'
                            stats['error'] = '第一列为空'
                            problem_files.append(stats)
                    else:
                        stats['status'] = '格式错误'
                        stats['error'] = '无法分割列'
                        problem_files.append(stats)
                else:
                    stats['status'] = '空文件'
                    stats['error'] = '第一行为空'
                    problem_files.append(stats)
        
        except Exception as e:
            stats['status'] = '读取错误'
            stats['error'] = str(e)
            problem_files.append(stats)
        
        file_stats.append(stats)
    
    # 分析结果
    print(f"\n诊断结果:")
    print(f"  总文件数: {len(tsv_files)}")
    print(f"  有效文件数: {valid_files}")
    print(f"  问题文件数: {len(problem_files)}")
    print(f"  唯一UniProt ID数: {len(uniprot_id_counter)}")
    
    # 检查重复的UniProt ID
    duplicate_ids = {id: count for id, count in uniprot_id_counter.items() if count > 1}
    
    if duplicate_ids:
        print(f"\n发现重复的UniProt ID:")
        for uniprot_id, count in duplicate_ids.items():
            print(f"  {uniprot_id}: 出现 {count} 次")
            
            # 找出包含这个ID的文件
            duplicate_files = []
            for stats in file_stats:
                if stats['uniprot_id'] == uniprot_id:
                    duplicate_files.append(stats['file_name'])
            
            print(f"    出现在: {', '.join(duplicate_files)}")
    
    # 显示问题文件
    if problem_files:
        print(f"\n问题文件详情:")
        
        # 按错误类型分组
        error_groups = {}
        for stats in problem_files:
            error_type = stats['status']
            if error_type not in error_groups:
                error_groups[error_type] = []
            error_groups[error_type].append(stats)
        
        for error_type, files in error_groups.items():
            print(f"  {error_type} ({len(files)}个):")
            for i, stats in enumerate(files[:5]):  # 最多显示5个
                print(f"    {i+1}. {stats['file_name']} - {stats.get('error', '')}")
            if len(files) > 5:
                print(f"    ... 还有 {len(files) - 5} 个")
    
    # 保存诊断报告
    save_diagnosis_report(file_stats, duplicate_ids, tsv_files)
    
    return file_stats, duplicate_ids, uniprot_id_counter

def save_diagnosis_report(file_stats, duplicate_ids, tsv_files):
    """
    保存详细的诊断报告
    """
    # 创建DataFrame
    stats_df = pd.DataFrame(file_stats)
    stats_df.to_csv("file_diagnosis_report.csv", index=False)
    
    # 创建文本报告
    with open("file_diagnosis_summary.txt", "w") as f:
        f.write("="*60 + "\n")
        f.write("文件诊断报告\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"总文件数: {len(tsv_files)}\n")
        f.write(f"有效文件数: {len([s for s in file_stats if s['status'] == '正常'])}\n")
        f.write(f"问题文件数: {len([s for s in file_stats if s['status'] != '正常'])}\n")
        f.write(f"唯一UniProt ID数: {len(set([s['uniprot_id'] for s in file_stats if s['uniprot_id']]))}\n\n")
        
        # 重复ID报告
        if duplicate_ids:
            f.write(f"重复的UniProt ID (共 {len(duplicate_ids)} 个):\n")
            for uniprot_id, count in duplicate_ids.items():
                f.write(f"\n  {uniprot_id} (出现 {count} 次):\n")
                for stats in file_stats:
                    if stats['uniprot_id'] == uniprot_id:
                        f.write(f"    - {stats['file_name']}\n")
        
        # 问题文件报告
        problem_files = [s for s in file_stats if s['status'] != '正常']
        if problem_files:
            f.write(f"\n问题文件 (共 {len(problem_files)} 个):\n")
            for stats in problem_files:
                f.write(f"\n  {stats['file_name']}:\n")
                f.write(f"    状态: {stats['status']}\n")
                f.write(f"    错误: {stats.get('error', '无')}\n")
                f.write(f"    大小: {stats['file_size']} 字节\n")
                f.write(f"    行数: {stats['line_count']}\n")
    
    print(f"\n诊断报告已保存:")
    print(f"  - file_diagnosis_report.csv")
    print(f"  - file_diagnosis_summary.txt")

def debug_ipr_extraction(file_path, max_lines=5):
    """
    调试函数：查看文件中IPR的分布位置
    """
    print(f"\n调试文件: {os.path.basename(file_path)}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines[:max_lines]):
        parts = line.strip().split('\t')
        print(f"\n行 {i+1}: {len(parts)}列")
        
        # 查找IPR
        for col_idx, part in enumerate(parts):
            if 'IPR' in part:
                print(f"  列{col_idx}: '{part}' (包含IPR)")
            elif col_idx < 15:  # 只显示前15列
                if part and part != '-':
                    print(f"  列{col_idx}: '{part}'")
    
    return lines

def extract_ipr_from_line(parts):
    """
    从一行中提取IPR特征
    策略：检查每一列是否包含IPR
    """
    iprs = set()
    
    for col_idx, field in enumerate(parts):
        if not field or field == '-':
            continue
        
        # 查找IPR标识符
        # IPR通常以"IPR"开头，后跟数字，例如：IPR000719
        if field.startswith('IPR'):
            # 直接添加
            iprs.add(field)
        elif 'IPR' in field:
            # 可能包含多个IPR用|分隔
            for item in field.split('|'):
                item = item.strip()
                if item.startswith('IPR'):
                    iprs.add(item)
                elif 'IPR' in item:
                    # 进一步提取
                    ipr_matches = re.findall(r'IPR\d{6}', item)
                    iprs.update(ipr_matches)
    
    return iprs

# ==================== 修改: 增强extract_all_ipr_from_file函数 ====================
def extract_all_ipr_from_file(file_path, check_duplicates=True):
    """
    从单个文件中提取所有IPR特征
    """
    uniprot_id = None
    ipr_set = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 检查文件是否为空
        if not lines:
            print(f"  ⚠️ 文件为空: {os.path.basename(file_path)}")
            return None, set()
        
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) < 3:  # 跳过太短的行
                continue
            
            if uniprot_id is None:
                uniprot_id = parts[0].strip()
                if not uniprot_id:
                    print(f"  ⚠️ 无UniProt ID: {os.path.basename(file_path)}")
                    return None, set()
            
            # 提取IPR
            iprs = extract_ipr_from_line(parts)
            ipr_set.update(iprs)
        
        if ipr_set:
            print(f"  ✅ {uniprot_id}: 找到 {len(ipr_set)} 个IPR")
        else:
            print(f"  ⚠️ {uniprot_id}: 未找到IPR特征")
        
        return uniprot_id, ipr_set
    
    except Exception as e:
        print(f"  ❌ 读取文件 {os.path.basename(file_path)} 时出错: {e}")
        return None, set()

# ==================== 修改: 增强build_ipr_vocabulary函数 ====================
def build_ipr_vocabulary(folder_path, min_frequency=1):
    """
    构建IPR词汇表
    """
    tsv_files = glob.glob(os.path.join(folder_path, "*.tsv"))
    print(f"找到 {len(tsv_files)} 个TSV文件")
    
    if not tsv_files:
        print("错误: 未找到TSV文件!")
        return None, None
    
    # 先调试前2个文件
    print("\n调试前2个文件的结构:")
    for i in range(min(2, len(tsv_files))):
        debug_ipr_extraction(tsv_files[i])
    
    # 统计所有IPR
    ipr_counter = Counter()
    protein_ipr_dict = {}
    processed_files = 0
    failed_files = 0
    duplicate_ids = {}
    
    print(f"\n开始提取IPR特征...")
    for file_path in tsv_files:
        uniprot_id, ipr_set = extract_all_ipr_from_file(file_path)
        
        if uniprot_id:
            # 检查是否重复
            if uniprot_id in protein_ipr_dict:
                if uniprot_id not in duplicate_ids:
                    duplicate_ids[uniprot_id] = []
                duplicate_ids[uniprot_id].append(os.path.basename(file_path))
                print(f"  ⚠️ 发现重复的UniProt ID: {uniprot_id}")
                
                # 合并IPR特征（取并集）
                existing_iprs = protein_ipr_dict[uniprot_id]
                combined_iprs = existing_iprs.union(ipr_set)
                protein_ipr_dict[uniprot_id] = combined_iprs
                print(f"    合并IPR: {len(existing_iprs)} + {len(ipr_set)} = {len(combined_iprs)} 个唯一IPR")
            else:
                protein_ipr_dict[uniprot_id] = ipr_set
            
            ipr_counter.update(ipr_set)
            processed_files += 1
        else:
            failed_files += 1
    
    print(f"\n处理完成统计:")
    print(f"  总文件数: {len(tsv_files)}")
    print(f"  成功处理: {processed_files}")
    print(f"  处理失败: {failed_files}")
    print(f"  唯一蛋白数: {len(protein_ipr_dict)}")
    print(f"  重复蛋白数: {len(duplicate_ids)}")
    print(f"  发现 {len(ipr_counter)} 个唯一IPR标识")
    
    if duplicate_ids:
        print(f"\n重复的UniProt ID详情:")
        for uniprot_id, files in duplicate_ids.items():
            print(f"  {uniprot_id}: 出现在 {len(files)} 个文件")
    
    if len(ipr_counter) == 0:
        print("\n警告: 未发现任何IPR标识!")
        print("尝试使用备用方法...")
        return build_ipr_vocabulary_alternative(folder_path, min_frequency)
    
    # 过滤低频IPR
    filtered_iprs = [(ipr, count) for ipr, count in ipr_counter.items() if count >= min_frequency]
    filtered_iprs.sort(key=lambda x: x[1], reverse=True)  # 按频率排序
    
    # 创建词汇表
    vocabulary = []
    for i, (ipr, count) in enumerate(filtered_iprs):
        vocabulary.append({
            'index': i,
            'ipr_id': ipr,
            'frequency': count,
            'percentage': (count / len(protein_ipr_dict)) * 100
        })
    
    print(f"过滤后保留 {len(vocabulary)} 个IPR标识 (阈值={min_frequency})")
    
    return vocabulary, protein_ipr_dict

def build_ipr_vocabulary_alternative(folder_path, min_frequency=1):
    """
    备用方法：使用更宽松的IPR提取规则
    """
    tsv_files = glob.glob(os.path.join(folder_path, "*.tsv"))
    print(f"\n备用方法: 尝试更宽松的IPR提取...")
    
    ipr_counter = Counter()
    protein_ipr_dict = {}
    
    for file_path in tsv_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用正则表达式提取所有IPR
        ipr_matches = re.findall(r'IPR\d{6}', content)
        
        if ipr_matches:
            uniprot_id = None
            # 尝试获取UniProt ID
            lines = content.strip().split('\n')
            if lines:
                first_line = lines[0]
                parts = first_line.split('\t')
                if parts:
                    uniprot_id = parts[0].strip()
            
            if uniprot_id:
                ipr_set = set(ipr_matches)
                protein_ipr_dict[uniprot_id] = ipr_set
                ipr_counter.update(ipr_set)
                print(f"  {uniprot_id}: 找到 {len(ipr_set)} 个IPR")
    
    print(f"\n备用方法结果:")
    print(f"  处理的蛋白: {len(protein_ipr_dict)}")
    print(f"  发现的IPR: {len(ipr_counter)}")
    
    if len(ipr_counter) == 0:
        print("错误: 仍然未发现任何IPR标识!")
        return [], {}
    
    # 过滤低频IPR
    filtered_iprs = [(ipr, count) for ipr, count in ipr_counter.items() if count >= min_frequency]
    filtered_iprs.sort(key=lambda x: x[1], reverse=True)
    
    # 创建词汇表
    vocabulary = []
    for i, (ipr, count) in enumerate(filtered_iprs):
        vocabulary.append({
            'index': i,
            'ipr_id': ipr,
            'frequency': count,
            'percentage': (count / len(protein_ipr_dict)) * 100
        })
    
    return vocabulary, protein_ipr_dict

def save_ipr_vocabulary(vocabulary, output_csv="ipr_vocabulary.csv"):
    """
    保存IPR词汇表
    """
    if not vocabulary:
        print("错误: 词汇表为空!")
        return None
    
    vocab_df = pd.DataFrame(vocabulary)
    vocab_df.to_csv(output_csv, index=False)
    print(f"词汇表已保存到: {output_csv}")
    print(f"包含 {len(vocab_df)} 个IPR特征")
    print(f"前10个最常见的IPR:")
    for i, row in vocab_df.head(10).iterrows():
        print(f"  {i+1}. {row['ipr_id']}: {row['frequency']}次 ({row['percentage']:.1f}%)")
    
    return vocab_df

def create_ipr_matrix(protein_ipr_dict, vocabulary, output_csv="protein_ipr_matrix.csv"):
    """
    创建IPR二进制矩阵
    """
    if not vocabulary or not protein_ipr_dict:
        print("错误: 词汇表或蛋白数据为空!")
        return None
    
    # 创建IPR到索引的映射
    ipr_to_index = {item['ipr_id']: item['index'] for item in vocabulary}
    
    n_proteins = len(protein_ipr_dict)
    n_features = len(vocabulary)
    
    print(f"\n创建 {n_proteins} × {n_features} 的二进制矩阵")
    
    # 准备数据
    matrix_data = []
    uniprot_ids = []
    
    for uniprot_id, ipr_set in protein_ipr_dict.items():
        # 创建零向量
        vector = [0] * n_features
        
        # 为存在的IPR置1
        for ipr in ipr_set:
            if ipr in ipr_to_index:
                idx = ipr_to_index[ipr]
                vector[idx] = 1
        
        matrix_data.append(vector)
        uniprot_ids.append(uniprot_id)
    
    # 创建DataFrame
    ipr_columns = [item['ipr_id'] for item in vocabulary]
    df = pd.DataFrame(matrix_data, columns=ipr_columns)
    df.insert(0, 'uniprot_id', uniprot_ids)
    
    # 保存
    df.to_csv(output_csv, index=False)
    print(f"矩阵已保存到: {output_csv}")
    print(f"矩阵形状: {df.shape}")
    
    # 分析稀疏性
    feature_matrix = df.drop('uniprot_id', axis=1)
    total_elements = feature_matrix.shape[0] * feature_matrix.shape[1]
    nonzero_elements = feature_matrix.values.sum()
    sparsity = 1 - (nonzero_elements / total_elements)
    
    print(f"\n稀疏性分析:")
    print(f"  总元素数: {total_elements}")
    print(f"  非零元素数: {nonzero_elements}")
    print(f"  稀疏度: {sparsity:.4f} ({sparsity*100:.2f}%)")
    print(f"  平均每个蛋白的IPR数: {nonzero_elements / n_proteins:.2f}")
    print(f"  平均每个IPR的蛋白数: {nonzero_elements / n_features:.2f}")
    
    return df

def get_protein_ipr_vector(uniprot_id, matrix_df, vocab_df=None):
    """
    获取指定蛋白的IPR向量
    """
    if uniprot_id not in matrix_df['uniprot_id'].values:
        print(f"未找到蛋白: {uniprot_id}")
        return None
    
    # 获取行数据
    row = matrix_df[matrix_df['uniprot_id'] == uniprot_id].iloc[0]
    
    # 提取向量
    vector = row.drop('uniprot_id').values
    
    # 统计
    total_features = len(vector)
    active_features = sum(vector > 0)
    
    print(f"蛋白: {uniprot_id}")
    print(f"总IPR特征维度: {total_features}")
    print(f"活跃IPR特征数: {active_features} ({active_features/total_features*100:.1f}%)")
    
    # 显示活跃的IPR
    if active_features > 0 and vocab_df is not None:
        active_indices = np.where(vector > 0)[0]
        print(f"\n活跃的IPR特征:")
        
        # 获取对应的IPR ID
        active_iprs = []
        for idx in active_indices:
            ipr_id = matrix_df.columns[idx + 1]  # +1 跳过uniprot_id列
            active_iprs.append(ipr_id)
        
        for i, ipr in enumerate(active_iprs[:10]):  # 最多显示10个
            print(f"  {i+1}. {ipr}")
        
        if len(active_iprs) > 10:
            print(f"  ... 还有 {len(active_iprs) - 10} 个IPR")
    
    return vector

# ==================== 修改: main函数添加诊断功能 ====================
def main():
    # 设置你的TSV文件所在文件夹路径
    tsv_folder = "/home/zouyanling/Direction/MoPE_MOE2/MoPE_MOE/dta/kiba/domain处理/所有domain文件"
    
    print("="*60)
    print("IPR特征提取与二进制编码系统（增强诊断版）")
    print("="*60)
    
    # ==================== 新增: 先运行文件诊断 ====================
    print("\n第一步: 运行文件诊断...")
    file_stats, duplicate_ids, uniprot_id_counter = diagnose_files(tsv_folder)
    
    total_files = len(glob.glob(os.path.join(tsv_folder, "*.tsv")))
    unique_ids = len(uniprot_id_counter)
    
    print(f"\n诊断总结:")
    print(f"  文件总数: {total_files}")
    print(f"  唯一UniProt ID数: {unique_ids}")
    
    if total_files == unique_ids:
        print(f"  ✅ 文件数与唯一ID数匹配")
    elif total_files > unique_ids:
        diff = total_files - unique_ids
        print(f"  ⚠️ 文件数({total_files}) > 唯一ID数({unique_ids})，差异: {diff}")
        print(f"    可能原因:")
        print(f"      1. 有空文件或格式错误的文件")
        print(f"      2. 有重复的UniProt ID")
    
    # 设置最小频率阈值
    min_freq = 1  # 只保留出现至少1次的IPR
    
    # 1. 构建IPR词汇表
    print("\n第二步: 构建IPR词汇表...")
    vocabulary, protein_ipr_dict = build_ipr_vocabulary(tsv_folder, min_frequency=min_freq)
    
    if not vocabulary:
        print("错误: 无法构建IPR词汇表!")
        return
    
    # 2. 保存词汇表
    print("\n第三步: 保存IPR词汇表...")
    vocab_df = save_ipr_vocabulary(vocabulary)
    
    # 3. 创建IPR矩阵
    print("\n第四步: 创建IPR二进制矩阵...")
    matrix_df = create_ipr_matrix(protein_ipr_dict, vocabulary)
    
    # 4. 示例
    print("\n第五步: 示例分析...")
    if not matrix_df.empty:
        # 显示前5个蛋白的统计
        print(f"\n前5个蛋白的IPR统计:")
        for i in range(min(5, len(matrix_df))):
            uniprot_id = matrix_df.iloc[i]['uniprot_id']
            vector = matrix_df.iloc[i].drop('uniprot_id').values
            active_count = sum(vector > 0)
            print(f"  {i+1}. {uniprot_id}: {active_count}个IPR")
        
        # 详细分析第一个蛋白
        first_id = matrix_df.iloc[0]['uniprot_id']
        print(f"\n第一个蛋白的详细信息 ({first_id}):")
        get_protein_ipr_vector(first_id, matrix_df, vocab_df)
    
    print(f"\n{'='*60}")
    print("处理完成!")
    print(f"{'='*60}")
    print(f"输出文件:")
    print(f"  1. 文件诊断报告: file_diagnosis_report.csv")
    print(f"  2. 文件诊断摘要: file_diagnosis_summary.txt")
    print(f"  3. IPR词汇表: ipr_vocabulary.csv")
    print(f"  4. IPR二进制矩阵: protein_ipr_matrix.csv")
    print(f"\n统计信息:")
    print(f"  处理的蛋白数: {len(protein_ipr_dict)}")
    print(f"  IPR特征维度: {len(vocabulary)}")
    print(f"  矩阵形状: {matrix_df.shape if matrix_df is not None else 'N/A'}")
    
    # 保存处理摘要
    summary = {
        'total_files': total_files,
        'unique_proteins': len(protein_ipr_dict),
        'duplicate_ids': len(duplicate_ids),
        'ipr_features': len(vocabulary),
        'matrix_shape': matrix_df.shape if matrix_df is not None else (0, 0),
        'output_files': [
            'file_diagnosis_report.csv',
            'file_diagnosis_summary.txt',
            'ipr_vocabulary.csv',
            'protein_ipr_matrix.csv'
        ]
    }
    
    import json
    with open('processing_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n处理摘要已保存: processing_summary.json")

if __name__ == "__main__":
    main()