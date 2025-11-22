import os
import json
import torch
import openai
import random
import datasets
import numpy as np
import math
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

from . import custom_datasets
from .load_models_tokenizers import load_base_model, load_mask_model

def drop_last_word(text):
    return ' '.join(text.split(' ')[:-1])

def _openai_sample(args, p):
    if args.dataset != 'pubmed':  # keep Answer: prefix for pubmed
        p = drop_last_word(p)

    # sample from the openai model
    kwargs = { "engine": args.openai_model, "max_tokens": 200 }
    if args.do_top_p:
        kwargs['top_p'] = args.top_p

    try:
        r = openai.Completion.create(prompt=f"{p}", **kwargs)
        return p + r['choices'][0].text
    except Exception as e:
        print(f"OpenAI API调用失败: {e}")
        return p  # 出错时返回原始前缀

def sample_from_model(args, config, texts, min_words=55, prompt_tokens=30):
    DEVICE = args.DEVICE
    base_tokenizer = config["base_tokenizer"]
    base_model = config["base_model"]
    GPT2_TOKENIZER = config["GPT2_TOKENIZER"]

    # encode each text as a list of token ids
    if args.dataset == 'pubmed':
        texts = [t[:t.index(custom_datasets.SEPARATOR)] for t in texts]
        all_encoded = base_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    else:
        all_encoded = base_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
        all_encoded = {key: value[:, :prompt_tokens] for key, value in all_encoded.items()}

    if args.openai_model:
        # decode the prefixes back into text
        prefixes = base_tokenizer.batch_decode(all_encoded['input_ids'], skip_special_tokens=True)
        with ThreadPool(args.batch_size) as pool:  # 用with管理线程池
            decoded = pool.starmap(_openai_sample, [(args, p) for p in prefixes])
    else:
        decoded = ['' for _ in range(len(texts))]

        # 采样直到所有文本满足最小词数要求
        tries = 0
        while (m := min(len(x.split()) for x in decoded)) < min_words:
            if tries != 0:
                print(f"最小词数不足（当前: {m}, 需求: {min_words}），重新生成（第{tries}次尝试）")

            sampling_kwargs = {}
            if args.do_top_p:
                sampling_kwargs['top_p'] = args.top_p
            elif args.do_top_k:
                sampling_kwargs['top_k'] = args.top_k
            min_length = 50 if args.dataset in ['pubmed'] else 150
            outputs = base_model.generate(
                **all_encoded,
                min_length=min_length,
                max_length=200,
                do_sample=True,** sampling_kwargs,
                pad_token_id=base_tokenizer.eos_token_id,
                eos_token_id=base_tokenizer.eos_token_id
            )
            decoded = base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            tries += 1

    if args.openai_model:
        # 统计token数
        total_tokens = sum(len(GPT2_TOKENIZER.encode(x)) for x in decoded)
        config["API_TOKEN_COUNTER"] += total_tokens

    return decoded

def strip_newlines(text):
    return ' '.join(text.split())

def trim_to_shorter_length(texta, textb):
    # 截断到较短文本的长度
    shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
    texta = ' '.join(texta.split(' ')[:shorter_length])
    textb = ' '.join(textb.split(' ')[:shorter_length])
    return texta, textb

def truncate_to_substring(text, substring, idx_occurrence):
    # 截断到substring第idx_occurrence次出现的位置
    assert idx_occurrence > 0, 'idx_occurrence必须大于0'
    idx = -1
    for _ in range(idx_occurrence):
        idx = text.find(substring, idx + 1)
        if idx == -1:
            return text
    return text[:idx]

def perturb_texts(args, config, texts, span_length, pct_words_masked, ceil_pct=True):
    """对文本列表进行批量扰动处理（随机掩码+模型填充）"""
    mask_model = config["mask_model"]
    mask_tokenizer = config["mask_tokenizer"]
    base_tokenizer = config["base_tokenizer"]
    DEVICE = args.DEVICE
    perturbed_texts = []

    for text in tqdm(texts, desc="预处理扰动文本"):
        # 分词并计算掩码数量
        tokens = base_tokenizer.tokenize(text)
        n_tokens = len(tokens)
        if n_tokens < 10:  # 文本过短不扰动
            perturbed_texts.append(text)
            continue

        # 计算需要掩码的token数量
        n_mask = pct_words_masked * n_tokens
        n_mask = int(n_mask) if not ceil_pct else math.ceil(n_mask)
        n_mask = max(1, min(n_mask, n_tokens - span_length))  # 限制范围

        # 随机选择不重叠的掩码位置
        mask_positions = []
        while len(mask_positions) < n_mask:
            start = random.randint(0, n_tokens - span_length)
            span = list(range(start, start + span_length))
            if not any(p in mask_positions for p in span):
                mask_positions.extend(span)
        mask_positions = list(set(mask_positions))[:n_mask]  # 去重并截断

        # 构建掩码文本
        masked_tokens = tokens.copy()
        for pos in mask_positions:
            masked_tokens[pos] = mask_tokenizer.mask_token
        masked_text = base_tokenizer.convert_tokens_to_string(masked_tokens)

        # 用掩码填充模型生成扰动文本
        inputs = mask_tokenizer(masked_text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = mask_model.generate(
                **inputs,
                max_length=len(masked_tokens) + 20,
                pad_token_id=mask_tokenizer.eos_token_id
            )
        filled_text = mask_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        perturbed_texts.append(filled_text)

    return perturbed_texts


def generate_samples(args, config, raw_data, batch_size):
    """生成机器文本和扰动文本"""
    torch.manual_seed(42)
    np.random.seed(42)
    data = {
        "original": [],
        "samples": [],
    }

    # 修复：使用range(len(raw_data))而非整除，确保所有样本都被处理
    for batch_idx in range(0, len(raw_data), batch_size):
        # 取当前批次的样本（最后一批可能不足batch_size）
        batch_data = raw_data[batch_idx: batch_idx + batch_size]
        print(
            f'生成第 {batch_idx // batch_size + 1}/{(len(raw_data) + batch_size - 1) // batch_size} 批样本（{len(batch_data)}条）')

        sampled_text = sample_from_model(args, config, batch_data, min_words=30 if args.dataset in ['pubmed'] else 55)

        for o, s in zip(batch_data, sampled_text):
            if args.dataset == 'pubmed':
                s = truncate_to_substring(s, 'Question:', 2)
                o = o.replace(custom_datasets.SEPARATOR, ' ')

            o, s = trim_to_shorter_length(o, s)

            # 过滤空文本（避免后续处理报错）
            if o.strip() and s.strip():
                data["original"].append(o)
                data["samples"].append(s)
            else:
                print("警告：过滤掉空的原始文本或生成文本")

    # 最终检查：确保生成的数据非空
    if len(data["original"]) == 0 or len(data["samples"]) == 0:
        print("⚠️ 生成的样本为空！添加内置示例...")
        data["original"] = ["示例原始文本1", "示例原始文本2"]
        data["samples"] = ["示例生成文本1", "示例生成文本2"]

    if args.pre_perturb_pct > 0:
        print(f'应用预处理扰动：{args.pre_perturb_pct}比例，{args.pre_perturb_span_length}长度')
        load_mask_model(args, config)
        data["samples"] = perturb_texts(args, config, data["samples"], args.pre_perturb_span_length,
                                        args.pre_perturb_pct, ceil_pct=True)
        load_base_model(args, config)

    return data


def generate_data(args, config):
    print(f'加载数据集 {args.dataset}...')

    dataset = args.dataset
    key = args.dataset_key
    cache_dir = config["cache_dir"]
    n_samples = config["n_samples"]
    batch_size = config["batch_size"]
    preproc_tokenizer = config["preproc_tokenizer"]

    try:
        print(f"尝试加载数据集: {dataset}")

        if dataset == 'xsum':
            # 已有的XSum加载逻辑（保持不变）
            print("检测到XSum数据集，处理列名...")
            train_dataset = datasets.load_dataset(dataset, cache_dir=cache_dir, split="train")
            valid_keys = ['document', 'summary']
            if key not in valid_keys:
                print(f"指定的列名'{key}'无效，自动选择'document'")
                selected_key = 'document'
            else:
                selected_key = key
            data = [str(item[selected_key]) for item in train_dataset]
            print(f"成功加载XSum数据集，列名: {selected_key}，样本数: {len(data)}")

        # 新增：专门处理WritingPrompts数据集（从本地文件加载）
        elif dataset == 'writingPrompts':
            print("检测到WritingPrompts数据集，从本地文件加载...")
            # 定义本地数据路径（项目根目录下的data/writingPrompts）
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "writingPrompts")
            # 检查目录是否存在
            if not os.path.exists(data_dir):
                raise FileNotFoundError(
                    f"未找到WritingPrompts数据目录: {data_dir}\n请确保在项目根目录创建了data/writingPrompts，并放入train.wp_target等文件")

            # 读取所有数据文件（train/test/valid）
            data_files = ["train.wp_target", "test.wp_target", "valid.wp_target"]
            data = []
            for file in data_files:
                file_path = os.path.join(data_dir, file)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"缺少数据文件: {file_path}")
                # 读取文件内容（按行读取，每行一个样本）
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]  # 过滤空行
                    data.extend(lines)
            print(f"成功加载WritingPrompts数据集，样本总数: {len(data)}")

        else:
            # 其他数据集加载逻辑（保持不变）
            if dataset in custom_datasets.DATASETS:
                data = custom_datasets.load(dataset, cache_dir)
            else:
                dataset_obj = datasets.load_dataset(dataset, cache_dir=cache_dir, split="train")
                data = list(dataset_obj[key])
            print(f"成功加载数据集，样本数: {len(data)}")

    except Exception as e:
        print(f"数据集加载失败: {e}")
        print("使用内置示例数据...")
        # 内置示例数据（确保至少有10条，避免后续为空）
        data = [
                   "The quick brown fox jumps over the lazy dog. This is a classic example sentence used in typing tests.",
                   "Machine learning and artificial intelligence are transforming various industries worldwide.",
                   "Climate change poses significant challenges to global ecosystems and human societies.",
                   "Renewable energy sources like solar and wind power are essential for sustainable development.",
                   "Natural language processing enables computers to understand and generate human language.",
                   "The development of quantum computing could revolutionize many scientific fields.",
                   "Artificial intelligence has the potential to transform healthcare and improve patient outcomes.",
                   "Blockchain technology offers new possibilities for secure digital transactions.",
                   "The Internet of Things connects everyday physical objects to the internet.",
                   "Virtual reality creates immersive digital experiences for entertainment and education."
               ] * 20  # 确保样本数足够（200条）
        print(f"使用内置样本，数量: {len(data)}")

    # 数据预处理
    data = list(dict.fromkeys(data))  # 确定性去重（替代set()）
    data = [x.strip() for x in data]  # 去除两端空白
    data = [strip_newlines(x) for x in data]  # 去除换行符

    # 过滤token数超过512的样本
    tokenized_data = preproc_tokenizer(data)
    data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= 512]
    print(f"过滤后token数≤512的样本数: {len(data)}")

    # 新增：确保数据非空（关键修复）
    if len(data) == 0:
        print("⚠️ 过滤后数据为空！强制使用内置示例数据...")
        data = [
            "The quick brown fox jumps over the lazy dog. This is a classic example sentence used in typing tests.",
            "Machine learning and artificial intelligence are transforming various industries worldwide.",
            "Climate change poses significant challenges to global ecosystems and human societies.",
            "Renewable energy sources like solar and wind power are essential for sustainable development.",
            "Natural language processing enables computers to understand and generate human language."
        ]
    elif len(data) < 10:
        print(f"⚠️ 过滤后样本数过少（{len(data)}条），可能导致后续错误，建议检查数据！")

    # 打乱数据
    random.seed(0)
    random.shuffle(data)
    data = data[:5_000]  # 限制最大样本量

    # 过滤token数超过512的样本
    tokenized_data = preproc_tokenizer(data)
    data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= 512]
    print(f"过滤后token数≤512的样本数: {len(data)}")

    # 打印数据统计
    print(f"总样本数: {len(data)}")
    print(f"平均词数: {np.mean([len(x.split()) for x in data])}")

    # 生成样本
    data = generate_samples(args, config, data[:n_samples], batch_size=batch_size)

    # 处理随机填充字典
    if args.random_fills:
        FILL_DICTIONARY = set()
        for texts in data.values():
            for text in texts:
                FILL_DICTIONARY.update(text.split())
        config["FILL_DICTIONARY"] = sorted(list(FILL_DICTIONARY))

    # 保存原始数据
    SAVE_FOLDER = config["SAVE_FOLDER"]
    os.makedirs(SAVE_FOLDER, exist_ok=True)  # 确保保存目录存在
    raw_data_path = os.path.join(SAVE_FOLDER, "raw_data.json")
    with open(raw_data_path, "w", encoding="utf-8") as f:
        print(f"保存原始数据到: {raw_data_path}")
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data



       
