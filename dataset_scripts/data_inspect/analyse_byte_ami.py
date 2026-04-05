from sklearn.metrics import adjusted_mutual_info_score
from torch.utils.data import Dataset
import json
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List, Union

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 18


def string_list_to_arr(string_list, num_string=5, string_len=320, pad_val=0):
    arr = []
    for string in string_list[:num_string]:
        string = string.strip()
        if len(string) == 0:
            arr.append([pad_val] * string_len)
        else:
            tmp_arr = [int(x) for x in string.split()]
            tmp_arr += [pad_val] * (string_len - len(tmp_arr))
            arr.append(tmp_arr[:string_len])
    arr += [[pad_val] * string_len] * (num_string - len(arr))
    return list(itertools.chain(*arr))


class ByteDataset(Dataset):
    def __init__(self, path_list, num_packet=5, num_packet_byte=320, transform=None):
        for path in path_list:
            assert os.path.exists(path), f"File not found: {path}"
        self.transform = transform
        self.data = []
        for path in path_list:
            tmp_data = json.load(open(path))
            print(f"Load {len(tmp_data)} items from {path}")
            self.data += tmp_data
        for item in self.data:
            item["data"] = string_list_to_arr(item["data"], num_string=num_packet, string_len=num_packet_byte)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["data"], item["label"]
    
    def __str__(self):
        return super().__str__() + f" with {len(self.data)} items"


def build_dataset(data_splits, data_path, num_packet=5, num_packet_byte=320):
    if isinstance(data_splits, list):
        paths = []
        for data_split in data_splits:
            assert data_split in ["train", "test", "valid"]
            paths.append(os.path.join(data_path, f"data-{data_split}.json"))
    elif isinstance(data_splits, str):
        assert data_splits in ["train", "test", "valid"]
        paths = [os.path.join(data_path, f"data-{data_splits}.json")]
    else:
        raise ValueError("data_splits should be a string or a list of strings")
    dataset = ByteDataset(paths, # transform=transform,
                          num_packet=num_packet, 
                          num_packet_byte=num_packet_byte)
    print(dataset)
    return dataset


def compute_ami_scores(dataset: Union[str, List[str]], byte_len: int, top_k_plot: int=10, print_title: bool=True):
    if isinstance(dataset, str):
        dataset_path = dataset_str = dataset
    elif isinstance(dataset, list):
        dataset_path = "/".join(dataset)
        dataset_str = "-".join(dataset)
    else:
        raise ValueError("dataset should be a string or a list of strings")
    num_packet = 5
    header_bytes = 80
    payload_bytes = 240
    data_path = f"/root/Vim/dataset_mm_uni/{dataset_path}/uni_nb={num_packet}_ns=50_hb={header_bytes}_pb={payload_bytes}_debias=none"
    dataset_all = build_dataset(["train", "test", "valid"], data_path, 
                            num_packet=num_packet, 
                            num_packet_byte=header_bytes+payload_bytes)
    X, y = [], []
    for item in dataset_all:
        X.append(item[0])
        y.append(item[1])
    X = np.array(X)
    X = X.reshape(X.shape[0], -1, byte_len)
    X = np.sum(X, axis=2) # sum of bytes

    ami_scores = []
    for i in range(X.shape[1]):
        ami_scores.append(adjusted_mutual_info_score(y, X[:, i]))
    sorted_idx = np.argsort(ami_scores)[::-1]
    sorted_ami_scores = np.array(ami_scores)[sorted_idx]
    sorted_output = []
    for idx in sorted_idx:
        sorted_output.append((f"Byte {idx*byte_len+1}-{(idx+1)*byte_len}", ami_scores[idx]))

    save_dir = f"../data_analysis/byte_ami/{dataset_str}/byte_len={byte_len},pkt_len={header_bytes+payload_bytes}"
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/scores.json", "w") as f:
        json.dump(sorted_output, f, indent=2)
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(top_k_plot), sorted_ami_scores[:top_k_plot])
    plt.xticks(range(top_k_plot), [f"Byte {idx*byte_len+1}-{(idx+1)*byte_len}" 
                                   for idx in sorted_idx[:top_k_plot]], rotation=45)
    plt.xlabel("Byte Range")
    plt.ylabel("AMI Score")
    if print_title:
        plt.title(f"Top-{top_k_plot} AMI Scores of Bytes in {dataset_str}")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/top{top_k_plot}.pdf")
    plt.show()

    # plot the scores in heatmap
    ami_matrix = np.array(ami_scores).reshape(num_packet*2, -1)
    print(f"ami matrix shape: {ami_matrix.shape}")
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(ami_matrix, cmap="Blues", interpolation="nearest", aspect="auto")
    # y_major_ticks = ax.set_yticks(np.linspace(0, ami_matrix.shape[0]-1, num=num_packet+1))
    # ax.set_yticklabels(np.arange(num_packet+1))
    # y_minor_ticks = np.linspace(0, ami_matrix.shape[0]-1, num=(len(y_major_ticks)-1)*(ami_matrix.shape[0]//num_packet)+1)
    # ax.set_yticks(y_minor_ticks, minor=True)
    ax.set_yticks(np.arange(ami_matrix.shape[0]))
    ax.set_yticklabels([f"{i}" for i in range(1, ami_matrix.shape[0]+1)])

    x_step = 2
    ax.set_xticks(np.arange(0, ami_matrix.shape[1], x_step))
    ax.set_xticklabels([f"{i}" for i in range(1, ami_matrix.shape[1]+1, x_step)])
    ax.set_xticks(np.arange(ami_matrix.shape[1]), minor=True)
    # fig.colorbar(im, ax=ax, orientation="vertical", pad=0.01)
    fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.11)
    if print_title:
        ax.set_title(f"AMI Scores of Consecutive {byte_len} Bytes in {dataset_str}")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/heatmap.pdf")


if __name__ == "__main__":
    datasets = ["CICIoT2022", "CipherSpectrum", ["CrossNet2021", "ScenarioA"], 
                ["CrossNet2021", "ScenarioB"], "CSTNET-TLS1.3", "ISCXVPN2016", "USTC-TFC2016",]
    byte_lens = [2, 4]
    for dataset in datasets:
        for byte_len in byte_lens:
            print(f"Computing AMI scores for {dataset} with byte length {byte_len}")
            compute_ami_scores(dataset, byte_len, top_k_plot=10, print_title=True)
        print(f"Finished computing AMI scores for {dataset}")
