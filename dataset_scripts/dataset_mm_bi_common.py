import json
import random
import os
import binascii
import numpy as np
import scapy.all as scapy
from tqdm import tqdm
from typing import Callable, Union, List
import math
from functools import partial
import shutil


def get_packet_feature(packet: scapy.Packet, header_bytes: int, payload_bytes: int,
                       merge_packet_bytes: bool):
    ip = packet["IPv6"] if packet.haslayer("IPv6") else packet["IP"]
    header = (binascii.hexlify(bytes(ip))).decode()
    try:
        payload = (binascii.hexlify(bytes(packet['Raw']))).decode()
        header = header.replace(payload, '')
    except:
        payload = ''
    header_arr = [int(header[i:i + 2], 16) for i in 
                    range(0, min(len(header), header_bytes*2), 2)]
    payload_arr = [int(payload[i:i + 2], 16) for i in 
                    range(0, min(len(payload), payload_bytes*2), 2)]
    if merge_packet_bytes:
        header_arr += [0] * (header_bytes - len(header_arr))
        payload_arr += [0] * (payload_bytes - len(payload_arr))
    return header_arr, payload_arr


##! if ip masking is required, src and dst ip should be mapped to fixed values to keep direction 
def get_chunk_feature(packets: scapy.PacketList, # first_src_ip: str,
                    byte_packet_num: int,
                    header_bytes: int, payload_bytes: int,
                    merge_packet_bytes: bool):
    bytes_list = []
    for packet in packets[:byte_packet_num]:
        header_arr, payload_arr = get_packet_feature(packet, header_bytes, payload_bytes, merge_packet_bytes)
        if merge_packet_bytes:
            bytes_list.append(header_arr + payload_arr)
        else:
            bytes_list.append([header_arr, payload_arr])
    signed_sizes, intervals = [], []
    # local first src ip determined by this flow chunk
    first_src_ip = packets[0]["IPv6"].src if packets[0].haslayer("IPv6") else packets[0]["IP"].src
    for idx, packet in enumerate(packets):
        src_ip = packet["IPv6"].src if packet.haslayer("IPv6") else packet["IP"].src
        if src_ip == first_src_ip:
            signed_sizes.append(len(packet)) # + for forward direction
        else:
            signed_sizes.append(-len(packet)) # - for backward direction
        if idx == 0:
            intervals.append(0)
        else:
            intervals.append(round(float(packet.time - packets[idx - 1].time), 6)) # round to 6 decimal places
    assert len(signed_sizes) == len(intervals)
    return bytes_list, signed_sizes, intervals


def merge_2d_array(arr_list):
    return [" ".join(map(str, arr)) for arr in arr_list]

def merge_1d_array(arr_list):
    return " ".join(map(str, arr_list))

def merge_2x2d_array(arr_list):
    return [(" ".join(map(str, arr[0])), " ".join(map(str, arr[1]))) for arr in arr_list]

def get_flow_feature(pcap_filename: str, byte_packet_num: int,
                   seq_packet_num: int, multi_chunk: bool,
                   header_bytes: int, payload_bytes: int,
                   debias_funcs: Union[List[Callable], None],
                   merge_packet_bytes: bool):
    packets = scapy.rdpcap(pcap_filename)
    if debias_funcs is not None:
        for debias_func in debias_funcs:
            packets = debias_func(packets)
    chunks = []
    total_packets = len(packets)
    num_chunks = total_packets // seq_packet_num
    if not multi_chunk or num_chunks == 0: # not enough packets to form a chunk
        bytes_list, signed_sizes, intervals = get_chunk_feature(
            packets, byte_packet_num, header_bytes, payload_bytes, merge_packet_bytes)
        chunks.append({
            "data": merge_2d_array(bytes_list) if merge_packet_bytes else merge_2x2d_array(bytes_list),
            "signed_sizes": merge_1d_array(signed_sizes),
            "intervals": merge_1d_array(intervals),
            "num_packet": len(packets),
        })
    else: # discard the last incomplete chunk
        for i in range(num_chunks):
            start_idx = i * seq_packet_num
            end_idx = start_idx + seq_packet_num
            bytes_list, signed_sizes, intervals = get_chunk_feature(
                packets[start_idx:end_idx], byte_packet_num, header_bytes, payload_bytes, merge_packet_bytes)
            chunks.append({
                "data": merge_2d_array(bytes_list) if merge_packet_bytes else merge_2x2d_array(bytes_list),
                "signed_sizes": merge_1d_array(signed_sizes),
                "intervals": merge_1d_array(intervals),
                "num_packet": len(packets[start_idx:end_idx]),
            })
    return chunks


def pcap_to_ft_json(pcap_dict_path: str, save_dir: str, 
                byte_packet_num: int, seq_packet_num: int,
                multi_chunk: bool, # whether to split a flow into multiple chunks
                header_bytes: int, payload_bytes: int,
                debias_funcs: Union[List[Callable], None] = None,
                merge_packet_bytes: bool = True):
    with open(pcap_dict_path, "r") as f:
        pcap_dict = json.load(f)
    app_names = list(pcap_dict["test"].keys())
    name_to_idx = {app_name: idx for idx, app_name in enumerate(app_names)}
    for split, split_dict in pcap_dict.items():
        data_list = []
        for app_name, app_pcap_files in split_dict.items():
            for app_pcap_file in tqdm(app_pcap_files, desc=f"Processing {split} {app_name}"):
                try:
                    res_list = get_flow_feature(app_pcap_file, byte_packet_num, seq_packet_num,
                                    multi_chunk, header_bytes, payload_bytes, debias_funcs, merge_packet_bytes)
                    for res in res_list:
                        if res["num_packet"] > 0:
                            data_list.append({
                                "data": res["data"],
                                "signed_sizes": res["signed_sizes"],
                                "intervals": res["intervals"],
                                "num_packet": res["num_packet"],
                                "label": name_to_idx[app_name],
                                "name": app_name,
                                "pcap_file": app_pcap_file,
                            })
                except Exception as e:
                    print(f"Error processing {app_pcap_file}: {e}")
        print(f"Save {len(data_list)} samples for {split} split to {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/data-{split}.json", "w") as f:
            json.dump(data_list, f, indent=2)
    with open(f"{save_dir}/metadata.json", "w") as f:
        json.dump({
            "name_to_idx": name_to_idx,
        }, f, indent=2)


def pcap_to_pt_json(pcap_dirs: List[str], save_dir: str, 
                filter_func: Union[Callable, None], 
                byte_packet_num: int, seq_packet_num: int,
                multi_chunk: bool,
                header_bytes: int, payload_bytes: int,
                debias_funcs: Union[List[Callable], None] = None,
                merge_packet_bytes: bool = True):
    pcap_dict = {}
    for pcap_dir in pcap_dirs:
        apps = list(filter(lambda x: os.path.isdir(f"{pcap_dir}/{x}"), os.listdir(pcap_dir)))
        if filter_func is not None:
            apps = list(filter(filter_func, apps))
        for app in apps:
            pcap_dict[app] = [f"{pcap_dir}/{app}/{pcap_file}" for pcap_file in
                                os.listdir(f"{pcap_dir}/{app}") if pcap_file.endswith(".pcap")]
    name_to_idx = {app_name: idx for idx, app_name in enumerate(pcap_dict.keys())}
    data_list = []
    for app_name in pcap_dict:
        app_pcap_files = pcap_dict[app_name]
        for app_pcap_file in tqdm(app_pcap_files, desc=f"Processing {app_name}"):
            try:
                res_list = get_flow_feature(app_pcap_file, byte_packet_num, seq_packet_num,
                                        multi_chunk, header_bytes, payload_bytes, debias_funcs, merge_packet_bytes)
                for res in res_list:
                    if res["num_packet"] > 0:
                        data_list.append({
                            "data": res["data"],
                            "signed_sizes": res["signed_sizes"],
                            "intervals": res["intervals"],
                            "num_packet": res["num_packet"],
                            "label": name_to_idx[app_name],
                            "name": app_name,
                            "pcap_file": app_pcap_file,
                        })
            except Exception as e:
                print(f"Error processing {app_pcap_file}: {e}")
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/data.json", "w") as f:
        json.dump(data_list, f, indent=2)

