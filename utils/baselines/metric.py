import numpy as np
from sklearn import metrics

def get_roc_metrics(real_preds, sample_preds):
    """
    计算 ROC 曲线指标，添加错误处理
    """
    try:
        # 合并真实和生成文本的预测分数
        predictions = real_preds + sample_preds
        labels = [1] * len(real_preds) + [0] * len(sample_preds)

        # 检查数据是否有效
        if len(predictions) == 0 or len(labels) == 0:
            print("⚠️ 警告: ROC计算 - 预测或标签为空")
            return np.array([0, 1]), np.array([0, 1]), 0.5

        # 检查标签是否只有一种类别
        unique_labels = set(labels)
        if len(unique_labels) == 1:
            print(f"⚠️ 警告: ROC计算 - 标签只有一种类别: {unique_labels}")
            if 1 in unique_labels:
                return np.array([0, 1]), np.array([1, 1]), 1.0
            else:
                return np.array([0, 1]), np.array([0, 1]), 0.0

        # 计算 ROC 曲线
        fpr, tpr, _ = metrics.roc_curve(labels, predictions)
        roc_auc = metrics.auc(fpr, tpr)

        return fpr, tpr, roc_auc

    except Exception as e:
        print(f"❌ ROC计算错误: {e}")
        # 返回默认的ROC曲线（对角线）
        return np.array([0, 1]), np.array([0, 1]), 0.5

def get_precision_recall_metrics(real_preds, sample_preds):
    """
    计算 Precision-Recall 曲线指标，添加错误处理
    """
    try:
        # 合并真实和生成文本的预测分数
        predictions = real_preds + sample_preds
        labels = [1] * len(real_preds) + [0] * len(sample_preds)

        # 检查数据是否有效
        if len(predictions) == 0 or len(labels) == 0:
            print("⚠️ 警告: PR计算 - 预测或标签为空")
            return np.array([1, 0]), np.array([0, 1]), 0.5

        # 检查标签是否只有一种类别
        unique_labels = set(labels)
        if len(unique_labels) == 1:
            print(f"⚠️ 警告: PR计算 - 标签只有一种类别: {unique_labels}")
            if 1 in unique_labels:
                return np.array([1, 1]), np.array([1, 0]), 1.0
            else:
                return np.array([1, 0]), np.array([0, 0]), 0.0

        # 计算 Precision-Recall 曲线
        precision, recall, _ = metrics.precision_recall_curve(labels, predictions)
        pr_auc = metrics.auc(recall, precision)

        return precision, recall, pr_auc

    except Exception as e:
        print(f"❌ PR计算错误: {e}")
        # 返回默认的PR曲线
        return np.array([1, 0]), np.array([0, 1]), 0.5
