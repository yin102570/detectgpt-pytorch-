import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 最小化实现，仅保证不报错

import torch
def get_lls(args, config, texts):
    """计算一组文本的对数似然"""
    base_model = config["base_model"]
    base_tokenizer = config["base_tokenizer"]
    DEVICE = args.DEVICE

    lls = []
    for text in texts:
        with torch.no_grad():
            tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
            labels = tokenized['input_ids']
            outputs = base_model(**tokenized, labels=labels)
            lls.append(-outputs.loss.item())

    return lls

def get_ll(args, config, text):
    """计算单个文本的对数似然"""
    return get_lls(args, config, [text])[0]



class LikelihoodScorer:
    def __init__(self, args, config, L_samples=None):
        self.args = args
        self.config = config
        self.L_samples = L_samples

    def score(self, text):
        """计算单个文本的对数似然分数"""
        return get_ll(self.args, self.config, text)

    def score_texts(self, texts):
        """计算一组文本的对数似然分数"""
        return [self.score(text) for text in texts]

class PerturbationScorer:
    def __init__(self, args, config, mask_filling_model, mask_filling_tokenizer):
        self.args = args
        self.config = config
        self.mask_filling_model = mask_filling_model
        self.mask_filling_tokenizer = mask_filling_tokenizer
        self.base_model = config["base_model"]
        self.base_tokenizer = config["base_tokenizer"]
        self.DEVICE = args.DEVICE

    def _perturb_text(self, text):
        """生成扰动文本（核心逻辑：随机掩码+模型填充）"""
        tokens = self.base_tokenizer.tokenize(text)
        n_tokens = len(tokens)
        if n_tokens < 10:
            return text  # 文本过短不扰动
        # 计算需要掩码的token数量（按pct_words_masked参数）
        n_mask = max(1, int(n_tokens * self.args.pct_words_masked))
        # 随机选择掩码位置（避免重叠）
        mask_positions = []
        while len(mask_positions) < n_mask:
            start = random.randint(0, n_tokens - self.args.span_length)
            span = list(range(start, start + self.args.span_length))
            if not any(p in mask_positions for p in span):
                mask_positions.extend(span)
        # 构建掩码文本
        masked_tokens = tokens.copy()
        for pos in mask_positions:
            masked_tokens[pos] = self.mask_filling_tokenizer.mask_token
        masked_text = self.base_tokenizer.convert_tokens_to_string(masked_tokens)
        # 用T5模型填充掩码
        inputs = self.mask_filling_tokenizer(masked_text, return_tensors="pt").to(self.DEVICE)
        with torch.no_grad():
            outputs = self.mask_filling_model.generate(**inputs, max_length=n_tokens + 20)
        filled_text = self.mask_filling_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return filled_text.strip()

    def score(self, text):
        """计算文本的扰动差异分数（原始文本LL - 扰动文本LL）"""
        # 计算原始文本对数似然
        original_ll = get_ll(self.args, self.config, text)
        # 生成扰动文本并计算LL（多次扰动取平均）
        perturbed_lls = []
        for _ in range(self.args.n_perturbation_rounds):
            perturbed_text = self._perturb_text(text)
            if perturbed_text and perturbed_text != text:
                perturbed_ll = get_ll(self.args, self.config, perturbed_text)
                perturbed_lls.append(perturbed_ll)
        if not perturbed_lls:
            return 0.0  # 无有效扰动时返回默认值
        avg_perturbed_ll = np.mean(perturbed_lls)
        # 扰动差异：原始LL越高、扰动LL越低，分数越大（区分人机文本）
        return original_ll - avg_perturbed_ll

    def score_texts(self, texts):
        """为文本列表计算分数（批量处理）"""
        return [self.score(text) for text in texts]

   