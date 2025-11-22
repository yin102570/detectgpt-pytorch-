import torch
import argparse
import json
import os
import sys

from utils.save_results import save_results
from utils.generate_data import generate_data
from utils.baselines.detectGPT import detectGPT
from utils.baselines.run_baselines import run_baselines
from utils.setting import set_experiment_config, initial_setup
from utils.load_models_tokenizers import load_base_model_and_tokenizer, load_base_model, load_mask_filling_model


def create_empty_results(output_dir):
    """创建空结果文件避免IndexError"""
    os.makedirs(output_dir, exist_ok=True)

    empty_files = {
        'baseline_outputs.json': [],
        'rank_threshold_results.json': {},
        'final_results.json': {}
    }

    for filename, content in empty_files.items():
        filepath = os.path.join(output_dir, filename)
        try:
            with open(filepath, 'w') as f:
                json.dump(content, f)
            print(f"✅ 创建空结果文件: {filepath}")
        except Exception as e:
            print(f"❌ 创建结果文件失败 {filepath}: {str(e)}")


def check_data_validity(data, min_samples=20):
    """检查数据有效性"""
    if not data or len(data) == 0:
        print("❌ 错误: 数据为空！")
        return False
    if len(data) < min_samples:
        print(f"⚠️ 警告: 样本数量不足 (需要≥{min_samples}，当前{len(data)})")
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 核心修改：将默认数据集从"xsum"改为"WritingPrompts"
    parser.add_argument('--dataset', type=str, default="WritingPrompts",
                        help="The dataset you want to run your experiments on. Natively supported: XSum, PubMedQA, WritingPrompts, SQuAD, English and German splits of WMT16.")
    # 核心修改：将默认数据集字段从"document"改为"prompt"
    parser.add_argument('--dataset_key', type=str, default="prompt",
                        help="The column of the dataset you want to use. For WritingPrompts, use 'prompt' or 'story'.")
    parser.add_argument('--pct_words_masked', type=float, default=0.3)
    parser.add_argument('--span_length', type=int, default=2,
                        help="Span lengths to mask for the mask filling model. A value of 2 performs the best.")
    parser.add_argument('--n_samples', type=int, default=200,
                        help="Number of samples to run the experiment on. For eg, if set to 200, will only use 200 and run all the experiments on them.")
    parser.add_argument('--n_perturbation_list', type=str, default="1,10",
                        help="Number of perturbed texts to generate to approximate the expectation term in the eq. 1 of the paper.")
    parser.add_argument('--n_perturbation_rounds', type=int, default=1, help="Rounds of perturbations to apply.")
    parser.add_argument('--base_model_name', type=str, default="gpt2-medium",
                        help="Base model to use to generate the machine-generated text.")
    parser.add_argument('--scoring_model_name', type=str, default="")
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-large",
                        help="Model to use for filling the masks.")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--chunk_size', type=int, default=20)
    parser.add_argument('--n_similarity_samples', type=int, default=20)
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--base_half', action='store_true')
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40,
                        help="Used for decoding strategy while generating the machine-generated text.")
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96,
                        help="Used for decoding strategy while generating the machine-generated text.")
    parser.add_argument('--output_name', type=str, default="")
    parser.add_argument('--openai_model', type=str, default=None)
    parser.add_argument('--openai_key', type=str)
    parser.add_argument('--baselines_only', action='store_true')
    parser.add_argument('--skip_baselines', action='store_true')
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--pre_perturb_pct', type=float, default=0.0)
    parser.add_argument('--pre_perturb_span_length', type=int, default=5)
    parser.add_argument('--random_fills', action='store_true')
    parser.add_argument('--random_fills_tokens', action='store_true')
    parser.add_argument('--cache_dir', type=str, default="cache")
    parser.add_argument('--DEVICE', type=str, default='cuda', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    config = {}

    try:
        # 解析命令行参数并设置实验
        initial_setup(args, config)
        set_experiment_config(args, config)

        # 加载基础模型和掩码填充模型
        load_base_model_and_tokenizer(args, config, None)
        load_mask_filling_model(args, config)

        # 将模型移动到DEVICE
        load_base_model(args, config)

        # 加载数据集并生成数据 (优化WritingPrompts处理)
        print(f"正在加载数据集: {args.dataset}")
        print(f"使用数据集字段: {args.dataset_key}")
        data = generate_data(args, config)

        # WritingPrompts数据集特殊处理
        if args.dataset.lower() == "writingprompts":
            print("检测到WritingPrompts数据集，应用特殊处理...")
            # WritingPrompts可能有较长的样本，调整过滤条件
            if not check_data_validity(data, min_samples=15):
                print("WritingPrompts样本不足，尝试使用story字段...")
                # 尝试使用story字段作为备选
                args.dataset_key = "story"
                print(f"重新使用数据集字段: {args.dataset_key}")
                data = generate_data(args, config)

                if not check_data_validity(data, min_samples=15):
                    print("❌ WritingPrompts数据集仍然不足，创建空结果文件并退出")
                    create_empty_results(config["output_dir"])
                    sys.exit(1)
        else:
            # 其他数据集的标准检查
            if not check_data_validity(data):
                print("尝试降低数据要求...")
                if not check_data_validity(data, min_samples=10):
                    print("❌ 数据不足，创建空结果文件并退出")
                    create_empty_results(config["output_dir"])
                    sys.exit(1)

        print(f"✅ 成功加载 {len(data)} 个有效样本")

        # 初始化输出容器
        baseline_outputs = []  # 初始化基线输出
        outputs = []  # 初始化DetectGPT输出

        # 如果指定了评分模型，先运行基线实验，然后切换模型
        if args.scoring_model_name:
            print(f'加载评分模型: {args.scoring_model_name}...')

            # 先运行基线实验（使用原始base_model）
            if not args.skip_baselines:
                if "base_model" in config:
                    print("运行基线实验...")
                    baseline_outputs = run_baselines(args, config, data)
                else:
                    print("⚠️ 警告: 基础模型未找到，跳过基线实验")

            # 清理原始模型并加载评分模型
            print("清理原始模型并加载评分模型...")
            if "base_model" in config:
                del config["base_model"]
            if "base_tokenizer" in config:
                del config["base_tokenizer"]

            # 强制释放GPU缓存
            if args.DEVICE == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("✅ GPU缓存已释放")

            # 加载评分模型
            load_base_model_and_tokenizer(args, config, args.scoring_model_name)
            load_base_model(args, config)
        else:
            # 如果没有指定评分模型，直接运行基线实验
            if not args.skip_baselines:
                if "base_model" in config:
                    print("运行基线实验...")
                    baseline_outputs = run_baselines(args, config, data)
                else:
                    print("⚠️ 警告: 基础模型未找到，跳过基线实验")

        # 运行DetectGPT算法
        if not args.baselines_only:
            if "base_model" in config:
                print("运行DetectGPT算法...")
                outputs = detectGPT(args, config, data, args.span_length)
            else:
                print("⚠️ 警告: 基础模型未加载！跳过DetectGPT实验")

        # 检查结果有效性 (关键修复点)
        if not baseline_outputs or len(baseline_outputs) == 0:
            print("⚠️ 警告: 基线输出为空，将创建空结果文件")
            create_empty_results(config["output_dir"])
            sys.exit(0)

        # 保存结果 (添加安全检查)
        try:
            save_results(args, config, baseline_outputs, outputs)
            print(f"✅ 结果已保存到: {config['output_dir']}")
        except IndexError as e:
            print(f"❌ 保存结果时发生索引错误: {str(e)}")
            print("尝试创建空结果文件...")
            create_empty_results(config["output_dir"])
        except Exception as e:
            print(f"❌ 保存结果时发生错误: {str(e)}")
            print("尝试创建空结果文件...")
            create_empty_results(config["output_dir"])

    except Exception as e:
        print(f"❌ 实验过程中发生错误: {str(e)}")
        # 尝试保存任何可能的结果
        if 'config' in locals() and 'output_dir' in config:
            print("尝试创建空结果文件...")
            create_empty_results(config["output_dir"])
        sys.exit(1)

