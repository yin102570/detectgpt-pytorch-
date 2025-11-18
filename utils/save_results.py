import os
import json
import numpy as np

def default_serializer(obj):
    """自定义 JSON 序列化函数，处理 numpy 类型"""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                       np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    else:
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

import matplotlib.pyplot as plt

# 15 colorblind-friendly colors
COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442",
            "#56B4E9", "#E69F00", "#000000", "#0072B2", "#009E73",
            "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]

def save_roc_curves(args, config, experiments):
    SAVE_FOLDER = config["SAVE_FOLDER"]
    base_model_name = config["base_model_name"]

    # first, clear plt
    plt.clf()

    for experiment, color in zip(experiments, COLORS):
        metrics = experiment["metrics"]
        plt.plot(metrics["fpr"], metrics["tpr"], label=f"{experiment['name']}, roc_auc={metrics['roc_auc']:.3f}", color=color)
        # print roc_auc for this experiment
        print(f"{experiment['name']} roc_auc: {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves ({base_model_name} - {args.mask_filling_model_name})')
    plt.legend(loc="lower right", fontsize=6)
    plt.savefig(f"{SAVE_FOLDER}/roc_curves.png")


# save the histogram of log likelihoods in two side-by-side plots, one for real and real perturbed, and one for sampled and sampled perturbed
def save_ll_histograms(args, config, experiments):
    SAVE_FOLDER = config["SAVE_FOLDER"]

    # first, clear plt
    plt.clf()

    for experiment in experiments:
        try:
            results = experiment["raw_results"]
            # plot histogram of sampled/perturbed sampled on left, original/perturbed original on right
            plt.figure(figsize=(20, 6))
            plt.subplot(1, 2, 1)
            plt.hist([r["sampled_ll"] for r in results], alpha=0.5, bins='auto', label='sampled')
            plt.hist([r["perturbed_sampled_ll"] for r in results], alpha=0.5, bins='auto', label='perturbed sampled')
            plt.xlabel("log likelihood")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.subplot(1, 2, 2)
            plt.hist([r["original_ll"] for r in results], alpha=0.5, bins='auto', label='original')
            plt.hist([r["perturbed_original_ll"] for r in results], alpha=0.5, bins='auto', label='perturbed original')
            plt.xlabel("log likelihood")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.savefig(f"{SAVE_FOLDER}/ll_histograms_{experiment['name']}.png")
        except:
            pass


# save the histograms of log likelihood ratios in two side-by-side plots, one for real and real perturbed, and one for sampled and sampled perturbed
def save_llr_histograms(args, config, experiments):
    SAVE_FOLDER = config["SAVE_FOLDER"]

    # first, clear plt
    plt.clf()

    for experiment in experiments:
        try:
            results = experiment["raw_results"]
            # plot histogram of sampled/perturbed sampled on left, original/perturbed original on right
            plt.figure(figsize=(20, 6))
            plt.subplot(1, 2, 1)

            # compute the log likelihood ratio for each result
            for r in results:
                r["sampled_llr"] = r["sampled_ll"] - r["perturbed_sampled_ll"]
                r["original_llr"] = r["original_ll"] - r["perturbed_original_ll"]
            
            plt.hist([r["sampled_llr"] for r in results], alpha=0.5, bins='auto', label='sampled')
            plt.hist([r["original_llr"] for r in results], alpha=0.5, bins='auto', label='original')
            plt.xlabel("log likelihood ratio")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.savefig(f"{SAVE_FOLDER}/llr_histograms_{experiment['name']}.png")
        except:
            pass

def save_results(args, config, baseline_outputs, outputs):
    SAVE_FOLDER = config["SAVE_FOLDER"]
    API_TOKEN_COUNTER = config["API_TOKEN_COUNTER"]

    if not args.skip_baselines:
        # write likelihood threshold results to a file
        with open(os.path.join(SAVE_FOLDER, f"likelihood_threshold_results.json"), "w") as f:
            json.dump(baseline_outputs, f, default=default_serializer, indent=2)

        if args.openai_model is None:
            # write rank threshold results to a file
            if len(baseline_outputs) >= 2:  # 确保列表至少有2个元素
                with open(os.path.join(SAVE_FOLDER, f"rank_threshold_results.json"), "w") as f:
                    json.dump(baseline_outputs[1], f)
            else:
                print("⚠️ baseline_outputs数据不足，跳过rank_threshold_results保存")
            

            # write log rank threshold results to a file
            with open(os.path.join(SAVE_FOLDER, f"logrank_threshold_results.json"), "w") as f:
                json.dump(baseline_outputs[2], f)

            # write entropy threshold results to a file
            with open(os.path.join(SAVE_FOLDER, f"entropy_threshold_results.json"), "w") as f:
                json.dump(baseline_outputs[3], f)
        
        # write supervised results to a file
        with open(os.path.join(SAVE_FOLDER, f"roberta-base-openai-detector_results.json"), "w") as f:
            json.dump(baseline_outputs[-2], f)
        
        # write supervised results to a file
        with open(os.path.join(SAVE_FOLDER, f"roberta-large-openai-detector_results.json"), "w") as f:
            json.dump(baseline_outputs[-1], f)

        outputs += baseline_outputs

    save_roc_curves(args, config, outputs)
    save_ll_histograms(args, config, outputs)
    save_llr_histograms(args, config, outputs)

    # move results folder from tmp_results/ to results/, making sure necessary directories exist
    new_folder = SAVE_FOLDER.replace("tmp_results", "results")
    if not os.path.exists(os.path.dirname(new_folder)):
        os.makedirs(os.path.dirname(new_folder))
    os.rename(SAVE_FOLDER, new_folder)

    print(f"Used an *estimated* {API_TOKEN_COUNTER} API tokens (may be inaccurate)")