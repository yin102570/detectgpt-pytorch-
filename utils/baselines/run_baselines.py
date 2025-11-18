import json
import numpy as np
from tqdm import tqdm
import torch

from .metric import get_roc_metrics, get_precision_recall_metrics
from .model import LikelihoodScorer, PerturbationScorer, get_ll



def run_baselines_threshold_experiment(args, data, criterion, name, L_samples=None):
    torch.manual_seed(0)
    np.random.seed(0)

    # 检查数据是否有效
    if len(data["original"]) == 0 or len(data["samples"]) == 0:
        print(f"⚠️ 警告: {name} 实验数据为空，跳过")
        return {
            "name": f"{name}_threshold",
            "predictions": {"real": [], "samples": []},
            "fpr": [],
            "tpr": [],
            "roc_auc": 0.5,
            "precision": [],
            "recall": [],
            "pr_auc": 0.5
        }

    # 使用 score_texts 方法计算分数
    real_pred = criterion.score_texts(data["original"])
    sampled_pred = criterion.score_texts(data["samples"])

    # 检查预测结果是否有效
    if len(real_pred) == 0 or len(sampled_pred) == 0:
        print(f"⚠️ 警告: {name} 预测结果为空，跳过")
        return {
            "name": f"{name}_threshold",
            "predictions": {"real": [], "samples": []},
            "fpr": [],
            "tpr": [],
            "roc_auc": 0.5,
            "precision": [],
            "recall": [],
            "pr_auc": 0.5
        }

    predictions = {
        "real": real_pred,
        "samples": sampled_pred,
    }

    # 计算 ROC 和 PR 指标
    try:
        fpr, tpr, roc_auc = get_roc_metrics(predictions["real"], predictions["samples"])
        precision, recall, pr_auc = get_precision_recall_metrics(predictions["real"], predictions["samples"])
    except Exception as e:
        print(f"⚠️ 计算指标时出错: {e}")
        # 返回默认值
        fpr, tpr, roc_auc = [0, 1], [0, 1], 0.5
        precision, recall, pr_auc = [1, 0], [0, 1], 0.5

    return {
        "name": f"{name}_threshold",
        "predictions": predictions,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "pr_auc": pr_auc,
    }

def run_baselines(args, config, data):
    # 检查数据是否有效
    if len(data["original"]) == 0 or len(data.get("samples", data.get("sampled", []))) == 0:
        print("⚠️ 警告: 输入数据为空，无法运行基线实验")
        return []

    L_samples = config.get("L_samples")  # 更安全的获取方式
    baseline_outputs = []

    # 1. Likelihood 实验
    try:
        likelihood_scorer = LikelihoodScorer(args, config)
        likelihood_output = run_baselines_threshold_experiment(
            args, data, likelihood_scorer, "likelihood", L_samples=L_samples
        )
        baseline_outputs.append(likelihood_output)
        print(f"✓ Likelihood 实验完成: AUC = {likelihood_output['roc_auc']:.3f}")
    except Exception as e:
        print(f"❌ Likelihood 实验失败: {e}")

    # 2. Perturbation 实验（仅当不跳过核心实验且不使用随机填充时运行）
    if not args.baselines_only and not args.random_fills:
        try:
            # 确保掩码模型和Tokenizer已加载并从config获取
            if "mask_model" not in config or "mask_tokenizer" not in config:
                from utils.load_models_tokenizers import load_mask_filling_model
                print("加载掩码填充模型和Tokenizer...")
                load_mask_filling_model(args, config)  # 加载并存入config

            # 初始化PerturbationScorer，传入必要参数
            perturbation_scorer = PerturbationScorer(
                args=args,
                config=config,
                mask_filling_model=config["mask_model"],
                mask_filling_tokenizer=config["mask_tokenizer"]
            )

            # 运行扰动实验
            perturbation_output = run_baselines_threshold_experiment(
                args, data, perturbation_scorer, "perturbation", L_samples=L_samples
            )
            baseline_outputs.append(perturbation_output)
            print(f"✓ Perturbation 实验完成: AUC = {perturbation_output['roc_auc']:.3f}")
        except Exception as e:
            print(f"❌ Perturbation 实验失败: {e}")

    return baseline_outputs

