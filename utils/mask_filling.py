
import torch
import random
import re
from transformers import T5ForConditionalGeneration, T5Tokenizer

class MaskFiller:
    def __init__(self, model_name, tokenizer, device):
        self.model = None  # 延迟加载
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.device = device

    def load_model(self):
        """延迟加载模型"""
        if self.model is None:
            print(f"加载掩码填充模型: {self.model_name}")
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

    def replace_masks(self, texts):
        """替换文本中的掩码"""
        self.load_model()

        # 简单的掩码替换逻辑
        replaced_texts = []
        for text in texts:
            # 查找 <extra_id_\d+> 模式的掩码
            masks = re.findall(r'<extra_id_\d+>', text)
            if masks:
                # 为每个掩码生成随机替换
                for mask in masks:
                    replacements = ["new", "different", "alternative", "modified", "changed"]
                    replacement = random.choice(replacements)
                    text = text.replace(mask, replacement, 1)
            replaced_texts.append(text)

        return replaced_texts

def perturb_texts(texts, pct=0.3, span_length=2, model_name="t5-small", tokenizer=None, device="cpu"):
    """
    扰动文本 - 简化版本
    """
    print(f"扰动 {len(texts)} 个文本，掩码比例: {pct}, 跨度长度: {span_length}")

    perturbed_texts = []
    mask_filler = MaskFiller(model_name, tokenizer, device)

    for text in texts:
        words = text.split()
        n_masks = max(1, int(len(words) * pct))

        # 创建掩码版本
        masked_text = text
        for i in range(n_masks):
            # 随机选择位置插入掩码
            if len(words) > span_length:
                start_pos = random.randint(0, len(words) - span_length)
                # 用掩码替换
                mask_token = f"<extra_id_{i}>"
                # 简化：在随机位置插入掩码
                words_with_mask = words.copy()
                insert_pos = random.randint(0, len(words_with_mask))
                words_with_mask.insert(insert_pos, mask_token)
                masked_text = " ".join(words_with_mask)

        # 替换掩码
        filled_texts = mask_filler.replace_masks([masked_text])
        perturbed_text = filled_texts[0] if filled_texts else text

        # 确保扰动后的文本与原始不同
        if perturbed_text == text:
            # 添加小的变化确保不同
            perturbed_text = text + " " if not text.endswith(" ") else text[:-1]

        perturbed_texts.append(perturbed_text)

    return perturbed_texts

if __name__ == "__main__":
    # 测试代码
    test_texts = ["Hello world this is a test.", "Another example sentence here."]
    result = perturb_texts(test_texts, pct=0.3, span_length=2)
    print("原始:", test_texts)
    print("扰动:", result)
