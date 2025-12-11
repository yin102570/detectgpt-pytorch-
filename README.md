# DetectGPT: 大语言模型生成文本检测项目
**Project Status**: 完成核心算法复现 & 全流程工程优化  
**Framework Support**: Python 3.12 | PyTorch 2.6.0 (cu126)  
**License**: MIT License


## 项目简介
本项目基于 ICML 2023 论文 **"DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature"** 及原官方仓库（[DetectGPT 官方仓库](https://github.com/the-utkarshjain/detectgpt)），在 PyTorch 框架下实现了 DetectGPT 算法的全流程复现与工程优化。项目整合了标准化数据集处理、可配置化实验流程、多维度性能评估工具，同时针对小样本场景新增自适应扰动、填充失败回退等机制，支持开发者快速复现算法核心逻辑并开展自定义检测任务。

### 核心价值
1. **零样本检测**: 无需标注数据训练分类器，仅通过语言模型概率曲率特性即可区分 AI 生成文本与人类文本，适配未见过的生成模型；  
2. **工程化适配**: 提供自定义数据集接口、动态参数配置、结果可视化工具，解决原始代码在实际场景中的易用性问题；  
3. **小样本优化**: 新增文本长度过滤、自适应扰动参数、填充失败回退策略，提升小样本场景下检测结果的稳定性；  
4. **全流程可复现**: 提供详细的环境配置脚本、实验运行指南、结果分析模板，确保算法逻辑与性能可复现。

### 算法核心原理
DetectGPT 的核心洞察是：**AI 生成文本在语言模型的概率空间中更"平滑"，人类文本更"曲折"**。  
- 对输入文本进行微小扰动（如随机掩码+T5模型填充同义词/短语），生成语义不变的扰动样本；  
- 计算原始文本与扰动样本在预训练语言模型中的对数似然值（概率得分）；  
- AI 文本的似然值在扰动后下降不明显（曲率低），人类文本的似然值下降显著（曲率高）；  
- 通过曲率阈值判断文本类型，实现零样本检测（流程示意图如下）：  
  ```
  输入文本 → 自适应扰动（生成k个样本）→ 似然值计算（原始+扰动样本）→ 曲率计算（原始似然-平均扰动似然）→ 阈值判断 → 输出AI/人类文本结果
  ```


## 项目架构
```
DetectGPT项目架构
├── 核心配置与入口
│   ├── run.py（实验主入口：参数解析、流程调度）
│   └── setting.py（配置层：输出目录创建、参数保存、环境初始化）
├── 核心算法模块
│   ├── detectGPT.py（DetectGPT核心逻辑：概率曲率计算、扰动-似然对比）
│   └── run_baselines.py（基线算法调度：调用baselines下的各类检测方法）
├── 工具支撑层
│   ├── model.py（模型工具：扰动生成、归一化似然计算、自适应参数调整）
│   ├── metric.py（评估工具：ROC-AUC计算、分类决策阈值判定）
│   └── utils/（工具子目录）
│       └── baselines/（基线算法实现）
│           ├── entropy.py（熵阈值检测算法）
│           ├── likelihood.py（似然阈值检测算法）
│           ├── rank.py（排序阈值检测算法）
│           └── supervised.py（有监督分类基线）
├── 数据与结果层
│   ├── generate_data.py（数据工具：数据集加载、过滤、回退机制）
│   ├── save_results.py（结果工具：结果存储、格式转换、可视化）
│   ├── data/（数据集目录：WritingPrompts等需手动下载）
│   ├── tmp_results/（中间结果目录：实验过程临时数据）
│   └── results/（最终结果目录：含args.json、各类阈值结果文件）
└── 辅助文件
    ├── .git/（Git版本管理目录）
    ├── __pycache__/（Python编译缓存目录）
    └── __init__.py（包初始化文件）



## 环境配置
### 硬件要求
| 设备类型 | 推荐配置 | 最低配置 | 备注 |
|----------|----------|----------|------|
| GPU | RTX 4090 (24GB) | RTX 3090 (24GB) | 运行 Llama-2-7B 等大模型需 24GB+ 显存 |
| CPU | AMD EPYC 7453 (14 vCPU) / Intel i7-13700K | Intel i5-12400 | 数据预处理与模型加载需多核心支持 |
| 内存 | 64 GB | 32 GB | 避免批量数据加载时内存溢出 |
| 硬盘 | 100 GB 空闲空间 | 50 GB 空闲空间 | 存储预训练模型（如 GPT-2 约 5GB，Llama-2-7B 约 13GB）与数据集 |

### 软件环境
| 依赖项 | 版本号 | 备注 |
|--------|--------|------|
| 操作系统 | Ubuntu 22.04 / Windows 10/11 (WSL2) | 建议使用 Ubuntu，避免 Windows 环境下的 CUDA 兼容问题 |
| Python | 3.12 | 需与依赖库版本匹配，建议用 Conda 隔离环境 |
| CUDA | 12.6 | 适配 PyTorch 2.6.0，需提前安装并配置环境变量 |
| CUDNN | 8.9.7.29 | 需匹配 CUDA 12.6，确保模型推理效率 |
| PyTorch | 2.6.0 (cu126) | 核心计算框架，选择与 CUDA 匹配的版本 |
| torchvision | 0.21.0 (cu126) | PyTorch 视觉依赖（本项目用于辅助数据处理） |
| torchaudio | 2.6.0 (cu126) | PyTorch 音频依赖（按需安装，非核心依赖） |
| transformers | 4.41.2 | 预训练模型（GPT/T5）加载与调用 |
| datasets | 2.19.0 | 公开数据集（XSum/SQuAD）自动加载 |
| scikit-learn | 1.4.2 | ROC-AUC、精确率/召回率等指标计算 |
| matplotlib | 3.8.4 | 实验结果可视化（ROC曲线、分布图） |

### 环境安装步骤
```bash
# 1. 创建并激活 Conda 虚拟环境（避免依赖冲突）
conda create -n detectgpt python=3.12 -y
conda activate detectgpt

# 2. 安装 PyTorch 及 CUDA 依赖（确保与 CUDA 12.6 匹配）
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# 3. 安装项目核心依赖（从 requirements.txt 读取，含版本锁定）
pip install -r requirements.txt

# 4. 验证环境配置（无报错则说明基础环境正常）
python -c "import torch; import transformers; 
print('PyTorch 版本:', torch.__version__); 
print('CUDA 可用:', torch.cuda.is_available()); 
print('Transformers 版本:', transformers.__version__)"
```

#### 常见环境问题解决
- **CUDA 版本不匹配**: 若执行 `torch.cuda.is_available()` 返回 `False`，需检查 CUDA 安装路径是否添加到环境变量，或重新安装与 CUDA 匹配的 PyTorch 版本。  
- **依赖安装缓慢**: 中国大陆用户可添加清华镜像源加速：  
  ```bash
  pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
  ```
- **T5 模型加载失败**: 确保 `transformers` 版本 ≥ 4.40.0，或手动下载模型文件放置到 `~/.cache/huggingface/hub/` 目录。


## 实验流程
### 1. 数据准备
#### 支持数据集与获取方式
| 数据集类型 | 适用场景 | 获取方式 | 存储路径 |
|------------|----------|----------|----------|
| WritingPrompts | 创意写作文本检测 | 从 Kaggle 手动下载：[WritingPrompts 数据集](https://www.kaggle.com/datasets/ratthachat/writing-prompts) | `./datasets/writingPrompts/` |
| XSum | 新闻摘要文本检测 | 自动加载（需 `datasets` 库） | 无需手动下载，运行时自动缓存 |
| SQuAD | 问答文本检测 | 自动加载（需 `datasets` 库） | 无需手动下载，运行时自动缓存 |
| 自定义数据集 | 特定领域文本检测 | 按格式整理为 TXT/JSON（每行1条文本） | 通过 `custom_datasets.py` 配置路径 |

#### 数据预处理（以 WritingPrompts 为例）
```bash
# 1. 手动下载 WritingPrompts 数据集并解压至 ./datasets/writingPrompts/
# 2. 运行预处理脚本，生成 100 个标准化样本（过滤极端长度、空文本）
python utils/generate_data.py \
    --dataset writingPrompts \
    --n_samples 100 \
    --output_path ./datasets/processed_writingPrompts.json
```

#### 小样本适配说明
- 若样本量不足（如 `n_samples < 50`），脚本会自动触发**回退机制**，补充占位文本避免实验结果随机；  
- 文本长度过滤规则：仅保留 10~500 词的文本，过滤过短（<10词）或过长（>500词）文本导致的似然值计算异常。


### 2. 实验运行
#### 主脚本核心命令（基础配置）
```bash
python run.py \
    --dataset writingPrompts \
    --base_model_name gpt2 \
    --mask_filling_model_name t5-small \
    --n_samples 100 \
    --n_perturbation_rounds 10 \
    --DEVICE cuda \
    --output_dir ./results/writingPrompts_gpt2_t5/
```

#### 关键参数说明
| 参数 | 含义 | 可选值 | 默认值 | 备注 |
|------|------|--------|--------|------|
| `--dataset` | 实验数据集名称 | writingPrompts/XSum/SQuAD/自定义名称 | writingPrompts | 自定义数据集需通过 `--custom_data_path` 指定路径 |
| `--base_model_name` | 基础语言模型（计算似然值） | gpt2/gpt-neo-1.3B/llama-2-7b | gpt2 | 大模型（如 llama-2-7b）需 24GB+ 显存 |
| `--mask_filling_model_name` | 扰动填充模型 | t5-small/t5-base | t5-small | t5-base 效果更优但显存占用更高 |
| `--n_samples` | 实验样本数量 | 正整数 | 100 | 小样本测试建议设为 10~20 |
| `--n_perturbation_rounds` | 每个文本的扰动轮次 | 正整数 | 10 | 轮次越多结果越稳定，但耗时更长 |
| `--DEVICE` | 运行设备 | cpu/cuda | cpu | 建议用 cuda 加速，cpu 推理耗时约为 cuda 的 20~30 倍 |
| `--output_dir` | 实验结果输出目录 | 自定义路径 | ./results/ | 自动创建目录，避免覆盖已有结果 |

#### 快速验证流程（10个样本，5轮扰动）
```bash
python run.py \
    --dataset writingPrompts \
    --n_samples 10 \
    --n_perturbation_rounds 5 \
    --DEVICE cuda \
    --output_dir ./results/quick_test/
```
- 该命令可在 5~10 分钟内完成全流程，用于验证环境配置与代码逻辑是否正常。


### 3. 结果分析
#### 输出文件说明（以 `./results/writingPrompts_gpt2_t5/` 为例）
| 文件名 | 内容说明 | 用途 |
|--------|----------|------|
| `detectgpt_results.json` | 原始结果：每个样本的文本内容、原始似然值、扰动似然值、曲率值、预测标签 | 追溯单样本检测细节 |
| `baseline_results.json` | 基线算法（似然阈值法）结果：每个样本的似然值、预测标签 | 与 DetectGPT 对比性能 |
| `metrics.csv` | 量化指标：AUC 分数、精确率、召回率、F1 分数（按算法分类） | 快速评估检测性能 |
| `roc_curve.png` | ROC 曲线对比图：DetectGPT 与似然阈值法的 TPR-FPR 曲线 | 可视化算法区分能力 |
| `ll_histogram.png` | 似然值分布图：人类文本与 AI 文本的似然值分布对比 | 分析两类文本的差异程度 |

#### 核心指标解读
- **AUC 分数**: 衡量算法区分能力，取值 0~1，越接近 1 性能越好。本项目在 WritingPrompts 数据集上，GPT-2+T5-small 配置下 AUC 约 0.78~0.797；  
- **曲率值**: 原始文本似然值 - 平均扰动似然值，差值越大，文本越可能是 AI 生成；  
- **ROC 曲线**: 曲线越靠近左上角，算法在低误判率（FPR）下的召回率（TPR）越高。


## 关键优化与复现经验
### 1. 工程化优化点
| 优化方向 | 具体实现 | 解决的问题 |
|----------|----------|------------|
| 自适应扰动 | 根据文本长度动态调整掩码比例（短文本 0.2、长文本 0.25）与 span 长度（1~3） | 避免短文本过度扰动（语义失真）或长文本扰动不足（差异不明显） |
| 填充失败回退 | T5 填充无效时保留原文片段并标记，而非直接丢弃扰动样本 | 解决小样本场景下有效扰动样本不足的问题 |
| 逐 token 归一化 | 似然值按文本 token 数量平均，而非直接使用总似然值 | 消除文本长度对似然值对比的影响 |
| 动态阈值选择 | 以所有样本曲率值的中位数为阈值，而非固定值 | 适配不同数据集的分布差异 |

### 2. 复现避坑指南
- **数据预处理细节**: 必须过滤含特殊符号（如 `\n`、`\t`）或极端长度的文本，否则会导致 T5 填充失败或似然值计算异常；  
- **显存不足处理**: 不要盲目缩减样本量（会导致结果随机），可通过以下方式优化：  
  1. 降低 `--n_samples` 为 20~50，分多批次运行后合并结果；  
  2. 使用小参模型（如 gpt2-small 替代 gpt2）；  
  3. 启用梯度 checkpointing（需在 `model.py` 中配置）；  
- **模型加载问题**: 若 Hugging Face 模型下载缓慢，可手动下载后放置到 `~/.cache/huggingface/hub/`，或配置镜像源：  
  ```bash
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_CACHE=/path/to/your/model/cache
  ```
- **结果稳定性**: 小样本（<50）实验建议重复运行 3~5 次，取 AUC 平均值，避免随机扰动导致的结果波动。


## 相关工作与引用
### 1. 参考项目与论文
- **原论文**: Mitchell, E., et al. (2023). DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature. ICML 2023.  
- **官方仓库**: [https://github.com/the-utkarshjain/detectgpt](https://github.com/the-utkarshjain/detectgpt)  
- **相关改进**: Fast-DetectGPT（西湖大学，2025）：检测速度提升 340 倍，AUC 相对提升约 75%，适用于大规模检测场景。

### 2. 论文引用格式（BibTeX）
```bibtex
@misc{mitchell2023detectgpt,
  url = {https://arxiv.org/abs/2301.11305},
  author = {Mitchell, Eric and Lee, Yoonho and Khazatsky, Alexander and Manning, Christopher D. and Finn, Chelsea},
  title = {DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature},
  publisher = {arXiv},
  year = {2023},
}
```


## 项目链接与联系方式
- **本项目仓库**: [https://github.com/yin102570/detectgpt-pytorch-](https://github.com/yin102570/detectgpt-pytorch-)  
- **问题反馈**: 直接在 GitHub Issues 提交问题


## License
本项目基于 MIT 协议开源，允许个人与商业使用，需保留原作者信息与协议声明。详情见项目根目录下的 `LICENSE` 文件。
