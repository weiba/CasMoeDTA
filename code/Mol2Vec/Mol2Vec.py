import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import argparse
import numpy as np
import pandas as pd

from rdkit import Chem
from tqdm import tqdm
from gensim.models import word2vec

from mol2vec.features import (
    mol2alt_sentence,
    MolSentence,
    Atom2Substructure
)


def get_wv_keys(model):
    """
    兼容 gensim 3.x 和 4.x
    """
    if hasattr(model.wv, "vocab"):
        return set(model.wv.vocab.keys())
    else:
        return set(model.wv.key_to_index.keys())


def get_word_vector(model, word):
    """
    兼容 gensim 3.x 和 4.x
    """
    if hasattr(model.wv, "word_vec"):
        return model.wv.word_vec(word)
    else:
        return model.wv[word]


def canonicalize_smiles(smiles):
    """
    将 SMILES 规范化，无法解析则返回 None
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--candidate_csv",
        type=str,
        default="chembl_approved_small_molecules_clean_not_in_davis.csv",
        help="ChEMBL 候选药物 CSV 文件，至少包含 chembl_id 和 canonical_smiles 两列"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="./model_300dim.pkl",
        help="Mol2Vec 预训练模型路径"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./chembl_candidate_features",
        help="输出特征文件目录"
    )

    parser.add_argument(
        "--id_col",
        type=str,
        default="chembl_id",
        help="候选药物 ID 列名"
    )

    parser.add_argument(
        "--smiles_col",
        type=str,
        default="canonical_smiles",
        help="候选药物 SMILES 列名"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("读取候选药物文件:", args.candidate_csv)
    df = pd.read_csv(args.candidate_csv)

    if args.id_col not in df.columns:
        raise ValueError(f"找不到 ID 列: {args.id_col}, 当前列名为: {df.columns.tolist()}")

    if args.smiles_col not in df.columns:
        raise ValueError(f"找不到 SMILES 列: {args.smiles_col}, 当前列名为: {df.columns.tolist()}")

    # 去除缺失 SMILES
    df = df[df[args.smiles_col].notna()].copy()

    # 去重
    df = df.drop_duplicates(subset=[args.id_col])

    print("候选药物数量:", len(df))

    print("加载 Mol2Vec 模型:", args.model_path)
    model = word2vec.Word2Vec.load(args.model_path)

    keys = get_wv_keys(model)

    unseen = "UNK"
    if unseen in keys:
        unseen_vec = get_word_vector(model, unseen)
    else:
        # 如果模型里没有 UNK，就用零向量作为未知子结构表示
        unseen_vec = np.zeros(model.vector_size, dtype=np.float32)
        print("Warning: 模型中未找到 UNK，使用零向量作为 unseen_vec")

    drug_descriptor = {}
    drug_matrix = {}

    failed_records = []
    too_small_records = []
    no_feature_records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating features"):
        drug_id = str(row[args.id_col])
        smiles = row[args.smiles_col]

        try:
            canonical_smiles = canonicalize_smiles(smiles)
            if canonical_smiles is None:
                failed_records.append({
                    "drug_id": drug_id,
                    "smiles": smiles,
                    "reason": "RDKit cannot parse SMILES"
                })
                continue

            mol = Chem.MolFromSmiles(canonical_smiles)
            if mol is None:
                failed_records.append({
                    "drug_id": drug_id,
                    "smiles": smiles,
                    "reason": "MolFromSmiles returns None"
                })
                continue

            sentence = MolSentence(mol2alt_sentence(mol, 1))

            # 注意：这里必须用你原代码的调用方式
            matrix = Atom2Substructure(mol, 1, model, keys, unseen_vec)
            matrix = np.asarray(matrix, dtype=np.float32)

            valid_words = [word for word in sentence if word in keys]

            if len(valid_words) > 0:
                vector = np.sum(
                    [get_word_vector(model, word) for word in valid_words],
                    axis=0
                ).astype(np.float32)
            else:
                if matrix.shape[0] > 0:
                    vector = np.mean(matrix, axis=0).astype(np.float32)
                    no_feature_records.append({
                        "drug_id": drug_id,
                        "smiles": canonical_smiles,
                        "reason": "No valid Mol2Vec words, use mean Atom2Vec matrix"
                    })
                else:
                    failed_records.append({
                        "drug_id": drug_id,
                        "smiles": canonical_smiles,
                        "reason": "No valid words and empty matrix"
                    })
                    continue

            if matrix.shape[0] <= 1:
                too_small_records.append({
                    "drug_id": drug_id,
                    "smiles": canonical_smiles,
                    "num_atoms": matrix.shape[0]
                })

            drug_descriptor[drug_id] = vector
            drug_matrix[drug_id] = matrix

        except Exception as e:
            failed_records.append({
                "drug_id": drug_id,
                "smiles": smiles,
                "reason": str(e)
            })

            
    mol2vec_path = os.path.join(args.output_dir, "compound_Mol2Vec300.pkl")
    atom2vec_path = os.path.join(args.output_dir, "compound_Atom2Vec300.pkl")

    with open(mol2vec_path, "wb") as f:
        pickle.dump(drug_descriptor, f)

    with open(atom2vec_path, "wb") as f:
        pickle.dump(drug_matrix, f)

    print("\n========== 生成完成 ==========")
    print("成功生成药物数量:", len(drug_descriptor))
    print("Mol2Vec 保存到:", mol2vec_path)
    print("Atom2Vec 保存到:", atom2vec_path)

    # 保存失败记录
    failed_df = pd.DataFrame(failed_records)
    too_small_df = pd.DataFrame(too_small_records)
    no_feature_df = pd.DataFrame(no_feature_records)

    failed_path = os.path.join(args.output_dir, "failed_drugs.csv")
    too_small_path = os.path.join(args.output_dir, "too_small_drugs.csv")
    no_feature_path = os.path.join(args.output_dir, "no_valid_words_drugs.csv")

    failed_df.to_csv(failed_path, index=False)
    too_small_df.to_csv(too_small_path, index=False)
    no_feature_df.to_csv(no_feature_path, index=False)

    print("失败药物数量:", len(failed_df))
    print("过小分子数量:", len(too_small_df))
    print("无有效 Mol2Vec word 但已用 matrix 均值替代的数量:", len(no_feature_df))

    print("失败记录保存到:", failed_path)
    print("过小分子记录保存到:", too_small_path)
    print("无有效词记录保存到:", no_feature_path)