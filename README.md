# DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature

重构自[DetectGPT paper](https://arxiv.org/abs/2301.11305v1)中的实验实现。

体验DetectGPT的交互式演示，请访问[此处](https://detectgpt.ericmitchell.ai).

## 安装说明

1. **安装Python依赖：**

   ```bash
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

2. **运行主程序：**

   使用以下命令启动主程序或`paper_scripts/`目录下的任意脚本：

   ```bash
   python run.py --base_model_name gpt2 --mask_filling_model_name t5-small --DEVICE cuda
   ```

3. **WritingPrompts实验：**

   若要运行WritingPrompts实验，需要从[这里](https://www.kaggle.com/datasets/ratthachat/writing-prompts)下载WritingPrompts数据，并将其保存至`data/writingPrompts`目录。

**注意：**实验的中间结果将保存在`tmp_results/`目录下。如果实验成功完成，结果会被移动至`results/`目录。

## 结果解释

成功运行脚本后，结果将被保存至`results/`目录。程序生成的文件及其说明如下：

1. **args.json**
   - 包含执行`main.py`时传递的命令行参数。

2. **entropy_threshold_results.json**
   - *predictions*字段存储真实文本与采样文本的熵值。*raw_results*字段为一个列表，包含每个样本的原始文本及其对应的熵值（*original_crit*），采样文本及其熵值（*sampled_crit*）。其他字段如*metrics*和*pr_metrics*未详细解释。

3. **likelihood_threshold_results.json**
   - 结构与上述文件相同，但值为平均对数似然。

4. **rank_threshold_results.json**
   - 结构与上述文件相同，但值为负排名值。

5. **logrank_threshold_results.json**
   - 结构与上述文件相同，但值为负对数排名值。

根据*n_perturbation_list*的配置，可能会生成下述文件：

1. **perturbation_1_d_results.json**
   - 存储未归一化的扰动差异值，每个结果包含原始文本、采样文本及其对数似然等信息。

2. **perturbation_1_z_results.json**
   - 存储归一化的扰动差异值。

3. perturbation_10_d_results.json
   - 与*perturbation_1_d_results.json*类似，但使用10个扰动样本。

4. perturbation_10_z_results.json
   - 与perturbation_1_z_results.json类似，但使用10个扰动样本。

## Citing the paper

If our work is useful for your own, you can cite us with the following BibTex entry:

@misc{mitchell2023detectgpt,

url = {https://arxiv.org/abs/2301.11305},

author = {Mitchell, Eric and Lee, Yoonho and Khazatsky, Alexander and Manning, Christopher D. and Finn, Chelsea},

title = {DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature},

publisher = {arXiv},

year = {2023},

}
