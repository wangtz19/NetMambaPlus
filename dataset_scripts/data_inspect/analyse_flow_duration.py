import os
import json
from tqdm import tqdm
from typing import Union
import scapy.all as scapy
import concurrent.futures
import matplotlib.pyplot as plt


def get_packet_timestamps(pcap_file):
    try:
        packets = scapy.rdpcap(pcap_file)
        if packets:
            return packets[0].time, packets[-1].time
        return float("inf"), float("inf")
    except Exception as e:
        return float("inf"), float("inf")


def count_pcap_duration(split_file: str, output_path: str):
    duration_dict = {"train": {}, "valid": {}, "test": {}}
    with open(split_file, "r") as f:
        split_data = json.load(f)
    for split in duration_dict.keys():
        for app_name, app_pcap_files in split_data[split].items():
            for app_pcap_file in tqdm(app_pcap_files, desc=f"Processing {split}/{app_name}"):
                start_time, end_time = get_packet_timestamps(app_pcap_file)
                if start_time == float("inf") or end_time == float("inf"):
                    continue
                duration = end_time - start_time
                if app_name not in duration_dict[split]:
                    duration_dict[split][app_name] = []
                duration_dict[split][app_name].append((float(start_time), float(end_time), float(duration)))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(duration_dict, f, indent=2)


def count_all_flows():
    tasks = []
    tasks.append(dict(
        split_file="/mnt/ssd1/wtz_nta_dataset/CICIoT2022/uni-flows_sampled_sorted.json",
        output_path="/root/Vim/dataset_scripts/data_analysis/flow_duration/CICIoT2022_sorted.json",
    ))
    tasks.append(dict(
        split_file="/mnt/ssd1/wtz_nta_dataset/CipherSpectrum/uni-flows-mix_sampled_sorted.json",
        output_path="/root/Vim/dataset_scripts/data_analysis/flow_duration/CipherSpectrum_sorted.json",
    ))
    tasks.append(dict(
        split_file="/mnt/ssd1/wtz_nta_dataset/CSTNET-TLS1.3/uni-flows_sampled_sorted.json",
        output_path="/root/Vim/dataset_scripts/data_analysis/flow_duration/CSTNET-TLS1.3_sorted.json",
    ))
    tasks.append(dict(
        split_file="/mnt/ssd1/wtz_nta_dataset/ISCXVPN2016/uni-flows_sampled_sorted.json",
        output_path="/root/Vim/dataset_scripts/data_analysis/flow_duration/ISCXVPN2016_sorted.json",
    ))
    tasks.append(dict(
        split_file="/mnt/ssd1/wtz_nta_dataset/USTC-TFC2016/uni-flows_sampled_sorted.json",
        output_path="/root/Vim/dataset_scripts/data_analysis/flow_duration/USTC-TFC2016_sorted.json",
    ))

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(count_pcap_duration, **task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in task: {e}")


def draw_flow_duration(duration_json: str, save_path: str):
    with open(duration_json, "r") as f:
        duration_dict = json.load(f)

    colors = {"train": "blue", "valid": "orange", "test": "green"}
    plt.figure(figsize=(16, 8))
    y = 0
    yticks = []
    ylabels = []

    for split in ["train", "valid", "test"]:
        for app_name, durations in duration_dict.get(split, {}).items():
            for start, end, dur in durations:
                plt.plot([start, end], [y, y], color=colors[split], linewidth=2)
                yticks.append(y)
                ylabels.append(f"{split}-{app_name}")
                y += 1

    plt.xlabel("Timestamp")
    plt.ylabel("Flow (split-app)")
    plt.title("Flow Duration Visualization")
    plt.yticks(yticks, ylabels, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()
    print(f"Saved flow duration plot to {save_path}")


if __name__ == "__main__":
    # count_all_flows()
    draw_flow_duration("/root/Vim/dataset_scripts/data_analysis/flow_duration/CICIoT2022_sorted.json", 
                       "/root/Vim/dataset_scripts/data_analysis/flow_duration/CICIoT2022_sorted.pdf")
    draw_flow_duration("/root/Vim/dataset_scripts/data_analysis/flow_duration/CipherSpectrum_sorted.json", 
                       "/root/Vim/dataset_scripts/data_analysis/flow_duration/CipherSpectrum_sorted.pdf")
    draw_flow_duration("/root/Vim/dataset_scripts/data_analysis/flow_duration/CSTNET-TLS1.3_sorted.json", 
                       "/root/Vim/dataset_scripts/data_analysis/flow_duration/CSTNET-TLS1.3_sorted.pdf")
    draw_flow_duration("/root/Vim/dataset_scripts/data_analysis/flow_duration/ISCXVPN2016_sorted.json",
                       "/root/Vim/dataset_scripts/data_analysis/flow_duration/ISCXVPN2016_sorted.pdf")
    draw_flow_duration("/root/Vim/dataset_scripts/data_analysis/flow_duration/USTC-TFC2016_sorted.json", 
                       "/root/Vim/dataset_scripts/data_analysis/flow_duration/USTC-TFC2016_sorted.pdf")
    # count_pcap_duration(split_file="/mnt/ssd1/wtz_nta_dataset/CipherSpectrum/uni-flows-mix_sampled_sorted.json",
    #     output_path="/root/Vim/dataset_scripts/data_analysis/flow_duration/CipherSpectrum_sorted.json",)