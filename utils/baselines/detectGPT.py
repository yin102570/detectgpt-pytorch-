
import numpy as np
from .model import PerturbationScorer

def detectGPT(args, config, data, span_length=2):
    
    print("运行简化版 DetectGPT...")

    # 确保数据键名正确
    if "samples" not in data:
        if "sampled" in data:
            data["samples"] = data["sampled"]

    original_texts = data["original"]
    sampled_texts = data["samples"]

    if len(original_texts) == 0 or len(sampled_texts) == 0:
        print("错误: 数据为空")
        return []
        # 获取扰动次数（修复作用域）
    n_perturbations = args.n_perturbation_list
        # 处理字符串格式的参数（如"1,10"），转为列表并取第一个值
    if isinstance(n_perturbations, str):
            n_perturbations = [int(x.strip()) for x in n_perturbations.split(",")][0]
    elif isinstance(n_perturbations, list):
            n_perturbations = n_perturbations[0]
    try:
        # 从config获取mask模型和tokenizer（关键依赖）
        mask_filling_model = config.get("mask_model")
        mask_filling_tokenizer = config.get("mask_tokenizer")
        if not mask_filling_model or not mask_filling_tokenizer:
            from utils.load_models_tokenizers import load_mask_filling_model
            load_mask_filling_model(args, config)
            mask_filling_model = config["mask_model"]
            mask_filling_tokenizer = config["mask_tokenizer"]
        scorer = PerturbationScorer(args, config, mask_filling_model, mask_filling_tokenizer)
        print("✅ 成功创建 PerturbationScorer")
    except Exception as e:
        print(f"创建评分器失败: {e}")
        return []
        
    

    # 计算分数
    print("计算原始文本分数...")
    original_scores = scorer.score_texts(original_texts)

    print("计算生成文本分数...")
    sampled_scores = scorer.score_texts(sampled_texts)

    # 准备结果
    results = {
        "name": f"perturbation_{n_perturbations}",
        "predictions": {
            "real": original_scores,
            "samples": sampled_scores
        },
        "info": {
            "pct_words_masked": args.pct_words_masked,
            "span_length": span_length,
            "n_perturbations": n_perturbations,
            "n_samples": len(original_texts)
        }
    }

    # 计算指标（简化版）
    try:
        from sklearn.metrics import roc_auc_score

        # 创建标签：原始文本为1，生成文本为0
        y_true = [1] * len(original_scores) + [0] * len(sampled_scores)
        y_scores = original_scores + sampled_scores

        if len(set(y_true)) > 1:  # 确保有正负样本
            auc = roc_auc_score(y_true, y_scores)
            results["roc_auc"] = auc
            print(f"DetectGPT AUC: {auc:.4f}")
        else:
            results["roc_auc"] = 0.5
            print("无法计算AUC（样本单一）")

    except Exception as e:
        print(f"计算指标失败: {e}")
        results["roc_auc"] = 0.5

    return [results]

# 为兼容性保留原有函数
def get_perturbation_results(args, config, data, span_length, n_perturbations, n_perturbation_rounds):
    """兼容性函数"""
    return detectGPT(args, config, data, span_length)
