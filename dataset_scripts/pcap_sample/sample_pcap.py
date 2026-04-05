import os
import random
import numpy as np
import json
from tqdm import tqdm
from typing import Union
import scapy.all as scapy


MAX_SAMPLES_PER_CLASS = 2000
MIN_SAMPLES_PER_CLASS = 50

def sample_pcap_files(input_dir: str, output_path: str, maximum: Union[int, float],
                      minimum, filter_func=None):
    app_names = list(filter(lambda x: os.path.isdir(f"{input_dir}/{x}") and len(os.listdir(f"{input_dir}/{x}")) >= minimum,
                            os.listdir(input_dir)))
    if filter_func is not None:
        app_names = list(filter(filter_func, app_names))
    split_dict = {"train": {}, "valid": {}, "test": {}}
    for app_name in app_names:
        pcap_files = list(filter(lambda x: x.endswith(".pcap"),
                                 os.listdir(f"{input_dir}/{app_name}")))
        pcap_files = [f"{input_dir}/{app_name}/{pcap_file}" for pcap_file in pcap_files]
        random.seed(0)
        random.shuffle(pcap_files)
        if 0 < maximum < 1:
            max_len = int(maximum * len(pcap_files))
        else:
            max_len = maximum
        pcap_files = pcap_files[:max_len] # Limit to maximum samples per class
        
        train_size = int(len(pcap_files) * 0.8)
        valid_size = int(len(pcap_files) * 0.1)
        # train 80%, valid 10%, test 10%
        
        split_dict["train"][app_name] = pcap_files[:train_size]
        split_dict["valid"][app_name] = pcap_files[train_size:train_size + valid_size]
        split_dict["test"][app_name] = pcap_files[train_size + valid_size:]
    with open(output_path, "w") as f:
        json.dump(split_dict, f, indent=2)


def get_first_packet_timestamp(pcap_file):
    try:
        packets = scapy.rdpcap(pcap_file, count=1)
        if packets:
            return packets[0].time
        return float("inf")
    except Exception as e:
        return float("inf")


def sample_pcap_files_by_time(input_dir: str, output_path: str, maximum: Union[int, float],
                      minimum, filter_func=None):
    app_names = list(filter(lambda x: os.path.isdir(f"{input_dir}/{x}") and len(os.listdir(f"{input_dir}/{x}")) >= minimum,
                            os.listdir(input_dir)))
    if filter_func is not None:
        app_names = list(filter(filter_func, app_names))
    split_dict = {"train": {}, "valid": {}, "test": {}}
    for app_name in app_names:
        pcap_files = list(filter(lambda x: x.endswith(".pcap"),
                                 os.listdir(f"{input_dir}/{app_name}")))
        pcap_files = [f"{input_dir}/{app_name}/{pcap_file}" for pcap_file in pcap_files]
        # sort pcap files by the first packet's timestamp in ascending order
        pcap_files.sort(key=lambda x: get_first_packet_timestamp(x))
        if 0 < maximum < 1:
            max_len = int(maximum * len(pcap_files))
        else:
            max_len = maximum
        pcap_files = pcap_files[:max_len] # Limit to maximum samples per class
        train_size = int(len(pcap_files) * 0.8)
        valid_size = int(len(pcap_files) * 0.1)
        # train 80%, valid 10%, test 10%
        split_dict["test"][app_name] = pcap_files[train_size + valid_size:]
        train_valid_files = pcap_files[:train_size + valid_size]
        # shuffle train and valid sets
        random.seed(0)
        random.shuffle(train_valid_files)
        split_dict["train"][app_name] = train_valid_files[:train_size]
        split_dict["valid"][app_name] = train_valid_files[train_size:train_size + valid_size]

    with open(output_path, "w") as f:
        json.dump(split_dict, f, indent=2)


def get_flow_name(granularity: str) -> str:
    if granularity == "session":
        return "bi-flows"
    elif granularity == "flow":
        return "uni-flows"
    else:
        raise ValueError("Granularity must be either 'session' or 'flow'.")


def sample_by_number(flow_name, dir_name="/mnt/ssd1/wtz_nta_dataset", type="normal"):
    datasets = [
        {"path": ["CICIoT2022", flow_name], "filter": None},
        {"path": ["CipherSpectrum", flow_name, "mix"], "filter": None},
        {"path": ["CrossNet2021", flow_name, "ScenarioA"], "filter": None},
        {"path": ["CrossNet2021", flow_name, "ScenarioB"], "filter": None},
        {"path": ["CSTNET-TLS1.3", flow_name], "filter": None},
        {"path": ["ISCXVPN2016", flow_name], "filter": lambda x: x.startswith("VPN")},
        {"path": ["USTC-TFC2016", flow_name], "filter": None},
        {"path": ["CrossPlatform", flow_name, "android"], "filter": None},
        {"path": ["CrossPlatform", flow_name, "ios"], "filter": None},
        {"path": ["DataCon2020-Malware", flow_name], "filter": None},
        {"path": ["DataCon2021-Proxy", "datacon2021_eta", "part1", flow_name], "filter": None},
        {"path": ["DataCon2021-Proxy", "datacon2021_eta", "part2", flow_name], "filter": None},
    ]
    for dataset in tqdm(datasets, desc="Sampling pcap files"):
        dataset_path = "/".join(dataset["path"])
        dataset_str = "-".join(dataset["path"])
        input_dir = f"{dir_name}/{dataset_path}"
        if type == "time":
            output_path = f"{dir_name}/{dataset['path'][0]}/{'-'.join(dataset['path'][1:])}_sampled_sorted.json"
        else:
            output_path = f"{dir_name}/{dataset['path'][0]}/{'-'.join(dataset['path'][1:])}_sampled.json"
        if "DataCon" in dataset_str:
            maximum = 500
        else:
            maximum = MAX_SAMPLES_PER_CLASS
        if type == "time":
            sample_pcap_files_by_time(input_dir, output_path, maximum, MIN_SAMPLES_PER_CLASS,
                                      filter_func=dataset["filter"])
        else:
            sample_pcap_files(input_dir, output_path, maximum, MIN_SAMPLES_PER_CLASS,
                            filter_func=dataset["filter"])


def sample_by_ratio(flow_name, dir_name="/mnt/ssd1/wtz_nta_dataset"):
    sample_pcap_files(
        input_dir=f"{dir_name}/CICIoT2022/{flow_name}",
        output_path=f"{dir_name}/CICIoT2022/{flow_name}_ratio_sampled.json",
        maximum=0.1,  # 10% of the original dataset
        minimum=MIN_SAMPLES_PER_CLASS,
        filter_func=None
    )
    sample_pcap_files(
        input_dir=f"{dir_name}/ISCXVPN2016/{flow_name}",
        output_path=f"{dir_name}/ISCXVPN2016/{flow_name}_ratio_sampled.json",
        maximum=0.4,  # 40% of the original dataset
        minimum=MIN_SAMPLES_PER_CLASS,
        filter_func=lambda x: x.startswith("VPN")
    )
    sample_pcap_files(
        input_dir=f"{dir_name}/USTC-TFC2016/{flow_name}",
        output_path=f"{dir_name}/USTC-TFC2016/{flow_name}_ratio_sampled.json",
        maximum=0.1,  # 10% of the original dataset
        minimum=MIN_SAMPLES_PER_CLASS,
        filter_func=None
    )


def sample_by_number_special(flow_name, dir_name="/mnt/ssd1/wtz_nta_dataset"):
    sample_pcap_files(
        input_dir=f"{dir_name}/CICIoT2022/{flow_name}",
        output_path=f"{dir_name}/CICIoT2022/{flow_name}_m6000_sampled.json",
        maximum=6000,
        minimum=MIN_SAMPLES_PER_CLASS,
        filter_func=None
    )
    sample_pcap_files(
        input_dir=f"{dir_name}/ISCXVPN2016/{flow_name}",
        output_path=f"{dir_name}/ISCXVPN2016/{flow_name}_m4000_sampled.json",
        maximum=4000,
        minimum=MIN_SAMPLES_PER_CLASS,
        filter_func=lambda x: x.startswith("VPN")
    )


if __name__ == "__main__":
    granularity = "flow"
    flow_name = get_flow_name(granularity)
    # sample_by_number(flow_name)
    # sample_by_ratio(flow_name)
    # sample_by_number_special(flow_name)
    # sample_by_number(flow_name, type="time")
    # sample_by_number(flow_name, type="time_adapt")
    sample_by_number(flow_name, type="time_adapt_iid")
