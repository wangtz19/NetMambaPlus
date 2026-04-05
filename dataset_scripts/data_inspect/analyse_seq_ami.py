import json
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Union


def str_to_arr(string: str, max_len: int, pad_value: float):
    arr = [float(x) for x in string.split(" ")]
    arr = arr[:max_len] + [pad_value] * (max_len - len(arr))
    return arr


def get_dataset(dataset_path: str, max_len: int, mtu: int,
                splits: List[str]=["train", "test", "valid"]):
    data = []
    for split in splits:
        assert split in ["train", "test", "valid"]
        data += json.load(open(f"/root/Vim/dataset_mm_uni/{dataset_path}/uni_nb=5_ns=50_hb=80_pb=240_debias=none/data-{split}.json"))
    res = []
    for item in data:
        sizes = str_to_arr(item["sizes"], max_len=max_len, pad_value=0)
        sizes = [0 if x < 0 else mtu if x > mtu else x for x in sizes]  # shift to [0, mtu]
        intervals = str_to_arr(item["intervals"], max_len=max_len, pad_value=float("inf"))
        res.append({
            "label": item["label"],
            "sizes": sizes,
            "intervals": intervals,
        })
    return res


def normalize_iat(iat: float, alpha: float=1.0):
    """
    iat_norm = sigmoid(alpha * log(1 + iat)) = 1 / (1 + exp(-alpha * log(1 + iat))) = 1 / (1 + 1 / (1 + iat) ** alpha)
    when iat = 0, iat_norm = 0.5;
    when iat -> inf, iat_norm -> 1;
    when iat -> -inf, iat_norm -> 0 (not applicable here since iat is non-negative).
    """
    return 1 / (1 + 1 / (1 + iat) ** alpha)


def build_dataset(data: List[dict], num_bins: int=50, mtu: int=1500,
                  features: List[str]=["sizes", "intervals"]):
    X, y = defaultdict(list), []
    for item in data:
        if any(len(item[feat]) < 2 for feat in features):
            continue
        for feat in features:
            map_fn = lambda x: normalize_iat(x) if feat == "intervals" else int(x)
            feat_values = list(map(map_fn, item[feat]))
            X[feat].append(feat_values)
        y.append(item["label"])

    for feat in features:
        if feat == "sizes":
            max_val, min_val = mtu, 0
        else: # intervals
            max_val, min_val = 1.0, 0.0
            for values in X[feat]:
                max_val = max(max_val, max(values))
                min_val = min(min_val, min(values))
        bins = np.linspace(min_val, max_val, num_bins)
        for i in range(len(X[feat])):
            X[feat][i] = np.digitize(X[feat][i], bins)
    return X, y


def compute_ami_scores(dataset: Union[str, List[str]], top_k_plot: int=20, num_bins: int=20, 
                       max_len: int=20, mtu: int=1500):
    if isinstance(dataset, str):
        dataset_path = dataset_str = dataset
    elif isinstance(dataset, list):
        dataset_path = "/".join(dataset)
        dataset_str = "-".join(dataset)
    else:
        raise ValueError("dataset should be a string or a list of strings")
    data = get_dataset(dataset_path, max_len=max_len, mtu=mtu)
    X_dict, y = build_dataset(data, num_bins=num_bins, mtu=mtu)
    sorted_feats = ["sizes", "intervals"]
    X = []
    for i in range(len(X_dict["sizes"])):
        val = []
        for feat in sorted_feats:
            val.extend(X_dict[feat][i])
        X.append(val)
    X = np.array(X) # shape: (num_samples, num_features*max_len)

    ami_scores = []
    for i in range(X.shape[1]):
        ami_scores.append(adjusted_mutual_info_score(y, X[:, i]))
    
    sorted_idx = np.argsort(ami_scores)[::-1]
    sorted_ami_scores = np.array(ami_scores)[sorted_idx]
    sorted_output = []

    def get_feat_name(idx):
        feat_idx = idx % max_len
        feat_name = sorted_feats[idx // max_len]
        return f"{feat_name}_{feat_idx}"

    for idx in sorted_idx:
        sorted_output.append((get_feat_name(idx), ami_scores[idx]))

    save_dir = f"../data_analysis/seq_ami/{dataset_str}//num_bins={num_bins},max_len={max_len}"
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/scores.json", "w") as f:
        json.dump(sorted_output, f, indent=2)
    
    plt.figure(figsize=(10, 5))
    _top_k_plot = min(top_k_plot, len(sorted_ami_scores))
    plt.bar(range(_top_k_plot), sorted_ami_scores[:_top_k_plot])
    plt.xticks(range(_top_k_plot), [get_feat_name(idx)
                                   for idx in sorted_idx[:_top_k_plot]], rotation=45)
    plt.xlabel("Feature")
    plt.ylabel("AMI Score")
    plt.title(f"Top-{_top_k_plot} AMI Scores of Features in {dataset_str}")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/top{_top_k_plot}.pdf")
    plt.show()

    # plot the scores in heatmap
    ami_matrix = np.array(ami_scores).reshape(len(sorted_feats), max_len)
    print(f"ami matrix shape: {ami_matrix.shape}")
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(ami_matrix, cmap="Blues", interpolation="nearest", aspect="auto")
    ax.set_yticks(np.arange(len(sorted_feats)))
    ax.set_yticklabels(sorted_feats)
    ax.set_xticks(np.arange(max_len))
    ax.set_xticklabels([f"{i}" for i in range(max_len)])
    fig.colorbar(im, ax=ax, orientation="vertical", pad=0.01)
    ax.set_title(f"AMI Scores of Features in {dataset_str}")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/heatmap.pdf")
    plt.show()


if __name__ == "__main__":
    datasets = ["CICIoT2022", "CipherSpectrum", ["CrossNet2021", "ScenarioA"], 
                ["CrossNet2021", "ScenarioB"], "CSTNET-TLS1.3", "ISCXVPN2016", "USTC-TFC2016",]
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        compute_ami_scores(dataset, top_k_plot=10, num_bins=20, max_len=20, mtu=1500)
    print("AMI scores computation completed for all datasets.")
