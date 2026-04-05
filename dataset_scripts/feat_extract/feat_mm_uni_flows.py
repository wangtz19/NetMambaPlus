import os
import sys
sys.path.append("/root/Vim/dataset_scripts")
import concurrent.futures


from dataset_debias_common import (
    zero_ip,
    zero_port,
    zero_ip_port,
    relative_zero_ip,
    relative_zero_port,
    zero_tls_sni,
    zero_tcp_window,
    zero_tcp_ts_option,
    relative_zero_tcp_ts_option,
    zero_ip_checksum,
    zero_transport_checksum,
    zero_seq_ack_no,
    relative_zero_seq_ack_no,
    zero_ip_ttl,
)
from dataset_mm_uni_common import pcap_to_pt_json, pcap_to_ft_json


debias_strategy = {
    "none": None,
    # "zero_ip": [zero_ip],
    # "zero_port": [zero_port],
    # "zero_ip_port": [zero_ip_port],
    # "zero_all_shortcut": [zero_ip_port, relative_zero_tcp_ts_option, relative_zero_seq_ack_no,
    #                       zero_ip_checksum, zero_transport_checksum]
}


BYTE_PACKET_NUM = 5
SEQ_PACKET_NUM = 50
HEADER_BYTES = 80
PAYLOAD_BYTES = 240
DATA_DIR_NAME = f"uni_nb={BYTE_PACKET_NUM}_ns={SEQ_PACKET_NUM}_hb={HEADER_BYTES}_pb={PAYLOAD_BYTES}"

dir_name = "/mnt/ssd1/wtz_nta_dataset/netx_data/dataset_mm_uni"
# dir_name = "/mnt/ssd2/wtz/ShieldGPT/datasets" # YaTC

def build_pt_data():
    for strategy in debias_strategy:
        DATA_PREFIX = f"{DATA_DIR_NAME}_debias={strategy}"
        # pcap_to_pt_json(
        #     pcap_dirs=["/mnt/ssd1/wtz_nta_dataset/Browser/uni-flows",
        #                "/mnt/ssd1/wtz_nids_dataset/kitsune/uni-flows"],
        #     save_dir=f"{dir_name}/pretrain-2/{DATA_PREFIX}",
        #     filter_func=None,
        #     byte_packet_num=BYTE_PACKET_NUM,
        #     seq_packet_num=SEQ_PACKET_NUM,
        #     multi_chunk=False,
        #     header_bytes=HEADER_BYTES,
        #     payload_bytes=PAYLOAD_BYTES,
        #     debias_funcs=debias_strategy[strategy],
        # )
        # pcap_to_pt_json(
        #     pcap_dirs=["/mnt/ssd1/wtz_nta_dataset/Browser/uni-flows",
        #                "/mnt/ssd1/wtz_nids_dataset/kitsune/uni-flows"],
        #     save_dir=f"{dir_name}/pretrain-non-merge/{DATA_PREFIX}",
        #     filter_func=None,
        #     byte_packet_num=BYTE_PACKET_NUM,
        #     seq_packet_num=SEQ_PACKET_NUM,
        #     multi_chunk=False,
        #     header_bytes=HEADER_BYTES,
        #     payload_bytes=PAYLOAD_BYTES,
        #     debias_funcs=debias_strategy[strategy],
        #     merge_packet_bytes=False,
        # )
        pcap_to_pt_json(
            pcap_dirs=["/mnt/ssd1/wtz_nta_dataset/Browser/uni-flows",
                       "/mnt/ssd1/wtz_nids_dataset/kitsune/uni-flows",
                       "/mnt/ssd1/wtz_nta_dataset/VNAT/uni-flows",],
            save_dir=f"{dir_name}/pretrain-3/{DATA_PREFIX}",
            filter_func=None,
            byte_packet_num=BYTE_PACKET_NUM,
            seq_packet_num=SEQ_PACKET_NUM,
            multi_chunk=False,
            header_bytes=HEADER_BYTES,
            payload_bytes=PAYLOAD_BYTES,
            debias_funcs=debias_strategy[strategy],
        )

def build_ft_data(max_workers=8):
    tasks = []
    for strategy in debias_strategy:
        DATA_PREFIX = f"{DATA_DIR_NAME}_debias={strategy}"
        # tasks.append(dict(
        #     pcap_dict_path="/mnt/ssd1/wtz_nta_dataset/ISCXVPN2016/uni-flows_new_app.json",
        #     save_dir=f"{dir_name}/ISCXVPN2016-new_app/{DATA_PREFIX}",
        #     byte_packet_num=BYTE_PACKET_NUM,
        #     seq_packet_num=SEQ_PACKET_NUM,
        #     multi_chunk=False,
        #     header_bytes=HEADER_BYTES,
        #     payload_bytes=PAYLOAD_BYTES,
        #     debias_funcs=debias_strategy[strategy],
        # ))
        # tasks.append(dict(
        #     pcap_dict_path="/mnt/ssd1/wtz_nta_dataset/ISCXVPN2016/uni-flows_varying_ratio.json",
        #     save_dir=f"{dir_name}/ISCXVPN2016-varying_ratio/{DATA_PREFIX}",
        #     byte_packet_num=BYTE_PACKET_NUM,
        #     seq_packet_num=SEQ_PACKET_NUM,
        #     multi_chunk=False,
        #     header_bytes=HEADER_BYTES,
        #     payload_bytes=PAYLOAD_BYTES,
        #     debias_funcs=debias_strategy[strategy],
        # ))
        # tasks.append(dict(
        #     pcap_dict_path="/mnt/ssd1/wtz_nta_dataset/CipherSpectrum/uni-flows_varying_cipher.json",
        #     save_dir=f"{dir_name}/CipherSpectrum-varying_cipher/{DATA_PREFIX}",
        #     byte_packet_num=BYTE_PACKET_NUM,
        #     seq_packet_num=SEQ_PACKET_NUM,
        #     multi_chunk=False,
        #     header_bytes=HEADER_BYTES,
        #     payload_bytes=PAYLOAD_BYTES,
        #     debias_funcs=debias_strategy[strategy],
        # ))
        tasks.append(dict(
            pcap_dict_path="/mnt/ssd1/wtz_nta_dataset/CICIoT2022/uni-flows_sampled_sorted.json",
            save_dir=f"{dir_name}/CICIoT2022-sorted/{DATA_PREFIX}",
            byte_packet_num=BYTE_PACKET_NUM,
            seq_packet_num=SEQ_PACKET_NUM,
            multi_chunk=False,
            header_bytes=HEADER_BYTES,
            payload_bytes=PAYLOAD_BYTES,
            debias_funcs=debias_strategy[strategy],
        ))
        tasks.append(dict(
            pcap_dict_path="/mnt/ssd1/wtz_nta_dataset/CipherSpectrum/uni-flows-mix_sampled_sorted.json",
            save_dir=f"{dir_name}/CipherSpectrum-sorted/{DATA_PREFIX}",
            byte_packet_num=BYTE_PACKET_NUM,
            seq_packet_num=SEQ_PACKET_NUM,
            multi_chunk=False,
            header_bytes=HEADER_BYTES,
            payload_bytes=PAYLOAD_BYTES,
            debias_funcs=debias_strategy[strategy],
        ))
        tasks.append(dict(
            pcap_dict_path="/mnt/ssd1/wtz_nta_dataset/CSTNET-TLS1.3/uni-flows_sampled_sorted.json",
            save_dir=f"{dir_name}/CSTNET-TLS1.3-sorted/{DATA_PREFIX}",
            byte_packet_num=BYTE_PACKET_NUM,
            seq_packet_num=SEQ_PACKET_NUM,
            multi_chunk=False,
            header_bytes=HEADER_BYTES,
            payload_bytes=PAYLOAD_BYTES,
            debias_funcs=debias_strategy[strategy],
        ))
        tasks.append(dict(
            pcap_dict_path="/mnt/ssd1/wtz_nta_dataset/ISCXVPN2016/uni-flows_sampled_sorted.json",
            save_dir=f"{dir_name}/ISCXVPN2016-sorted/{DATA_PREFIX}",
            byte_packet_num=BYTE_PACKET_NUM,
            seq_packet_num=SEQ_PACKET_NUM,
            multi_chunk=False,
            header_bytes=HEADER_BYTES,
            payload_bytes=PAYLOAD_BYTES,
            debias_funcs=debias_strategy[strategy],
        ))
        tasks.append(dict(
            pcap_dict_path="/mnt/ssd1/wtz_nta_dataset/USTC-TFC2016/uni-flows_sampled_sorted.json",
            save_dir=f"{dir_name}/USTC-TFC2016-sorted/{DATA_PREFIX}",
            byte_packet_num=BYTE_PACKET_NUM,
            seq_packet_num=SEQ_PACKET_NUM,
            multi_chunk=False,
            header_bytes=HEADER_BYTES,
            payload_bytes=PAYLOAD_BYTES,
            debias_funcs=debias_strategy[strategy],
        ))

        # tasks.append(dict(
        #     pcap_dict_path="/mnt/ssd1/wtz_nta_dataset/CICIoT2022/uni-flows_sampled_time_adapt_iid.json",
        #     save_dir=f"{dir_name}/CICIoT2022-taiid/{DATA_PREFIX}",
        #     byte_packet_num=BYTE_PACKET_NUM,
        #     seq_packet_num=SEQ_PACKET_NUM,
        #     multi_chunk=False,
        #     header_bytes=HEADER_BYTES,
        #     payload_bytes=PAYLOAD_BYTES,
        #     debias_funcs=debias_strategy[strategy],
        # ))
        # tasks.append(dict(
        #     pcap_dict_path="/mnt/ssd1/wtz_nta_dataset/CipherSpectrum/uni-flows-mix_sampled_time_adapt_iid.json",
        #     save_dir=f"{dir_name}/CipherSpectrum-taiid/{DATA_PREFIX}",
        #     byte_packet_num=BYTE_PACKET_NUM,
        #     seq_packet_num=SEQ_PACKET_NUM,
        #     multi_chunk=False,
        #     header_bytes=HEADER_BYTES,
        #     payload_bytes=PAYLOAD_BYTES,
        #     debias_funcs=debias_strategy[strategy],
        # ))
        # tasks.append(dict(
        #     pcap_dict_path="/mnt/ssd1/wtz_nta_dataset/CSTNET-TLS1.3/uni-flows_sampled_time_adapt_iid.json",
        #     save_dir=f"{dir_name}/CSTNET-TLS1.3-taiid/{DATA_PREFIX}",
        #     byte_packet_num=BYTE_PACKET_NUM,
        #     seq_packet_num=SEQ_PACKET_NUM,
        #     multi_chunk=False,
        #     header_bytes=HEADER_BYTES,
        #     payload_bytes=PAYLOAD_BYTES,
        #     debias_funcs=debias_strategy[strategy],
        # ))
        # tasks.append(dict(
        #     pcap_dict_path="/mnt/ssd1/wtz_nta_dataset/ISCXVPN2016/uni-flows_sampled_time_adapt_iid.json",
        #     save_dir=f"{dir_name}/ISCXVPN2016-taiid/{DATA_PREFIX}",
        #     byte_packet_num=BYTE_PACKET_NUM,
        #     seq_packet_num=SEQ_PACKET_NUM,
        #     multi_chunk=False,
        #     header_bytes=HEADER_BYTES,
        #     payload_bytes=PAYLOAD_BYTES,
        #     debias_funcs=debias_strategy[strategy],
        # ))
        # tasks.append(dict(
        #     pcap_dict_path="/mnt/ssd1/wtz_nta_dataset/USTC-TFC2016/uni-flows_sampled_time_adapt_iid.json",
        #     save_dir=f"{dir_name}/USTC-TFC2016-taiid/{DATA_PREFIX}",
        #     byte_packet_num=BYTE_PACKET_NUM,
        #     seq_packet_num=SEQ_PACKET_NUM,
        #     multi_chunk=False,
        #     header_bytes=HEADER_BYTES,
        #     payload_bytes=PAYLOAD_BYTES,
        #     debias_funcs=debias_strategy[strategy],
        # ))  

        tasks.append(dict(
            pcap_dict_path="/mnt/ssd1/wtz_nta_dataset/CICIoT2022/uni-flows_sampled.json",
            save_dir=f"{dir_name}/CICIoT2022/{DATA_PREFIX}",
            byte_packet_num=BYTE_PACKET_NUM,
            seq_packet_num=SEQ_PACKET_NUM,
            multi_chunk=False,
            header_bytes=HEADER_BYTES,
            payload_bytes=PAYLOAD_BYTES,
            debias_funcs=debias_strategy[strategy],
        ))
        tasks.append(dict(
            pcap_dict_path="/mnt/ssd1/wtz_nta_dataset/CipherSpectrum/uni-flows-mix_sampled.json",
            save_dir=f"{dir_name}/CipherSpectrum/{DATA_PREFIX}",
            byte_packet_num=BYTE_PACKET_NUM,
            seq_packet_num=SEQ_PACKET_NUM,
            multi_chunk=False,
            header_bytes=HEADER_BYTES,
            payload_bytes=PAYLOAD_BYTES,
            debias_funcs=debias_strategy[strategy],
        ))
        # for scenario in ["A", "B"]:
        #     tasks.append(dict(
        #         pcap_dict_path=f"/mnt/ssd1/wtz_nta_dataset/CrossNet2021/uni-flows-Scenario{scenario}_sampled.json",
        #         save_dir=f"{dir_name}/CrossNet2021/Scenario{scenario}/{DATA_PREFIX}",
        #         byte_packet_num=BYTE_PACKET_NUM,
        #         seq_packet_num=SEQ_PACKET_NUM,
        #         multi_chunk=False,
        #         header_bytes=HEADER_BYTES,
        #         payload_bytes=PAYLOAD_BYTES,
        #         debias_funcs=debias_strategy[strategy],
        #     ))
        tasks.append(dict(
            pcap_dict_path="/mnt/ssd1/wtz_nta_dataset/CSTNET-TLS1.3/uni-flows_sampled.json",
            save_dir=f"{dir_name}/CSTNET-TLS1.3/{DATA_PREFIX}",
            byte_packet_num=BYTE_PACKET_NUM,
            seq_packet_num=SEQ_PACKET_NUM,
            multi_chunk=False,
            header_bytes=HEADER_BYTES,
            payload_bytes=PAYLOAD_BYTES,
            debias_funcs=debias_strategy[strategy],
        ))
        tasks.append(dict(
            pcap_dict_path="/mnt/ssd1/wtz_nta_dataset/ISCXVPN2016/uni-flows_sampled.json",
            save_dir=f"{dir_name}/ISCXVPN2016/{DATA_PREFIX}",
            byte_packet_num=BYTE_PACKET_NUM,
            seq_packet_num=SEQ_PACKET_NUM,
            multi_chunk=False,
            header_bytes=HEADER_BYTES,
            payload_bytes=PAYLOAD_BYTES,
            debias_funcs=debias_strategy[strategy],
        ))
        tasks.append(dict(
            pcap_dict_path="/mnt/ssd1/wtz_nta_dataset/USTC-TFC2016/uni-flows_sampled.json",
            save_dir=f"{dir_name}/USTC-TFC2016/{DATA_PREFIX}",
            byte_packet_num=BYTE_PACKET_NUM,
            seq_packet_num=SEQ_PACKET_NUM,
            multi_chunk=False,
            header_bytes=HEADER_BYTES,
            payload_bytes=PAYLOAD_BYTES,
            debias_funcs=debias_strategy[strategy],
        ))
        # tasks.append(dict(
        #     pcap_dict_path="/mnt/ssd1/wtz_nta_dataset/CrossPlatform/uni-flows-android_sampled.json",
        #     save_dir=f"{dir_name}/CrossPlatform-Android/{DATA_PREFIX}",
        #     byte_packet_num=BYTE_PACKET_NUM,
        #     seq_packet_num=SEQ_PACKET_NUM,
        #     multi_chunk=False,
        #     header_bytes=HEADER_BYTES,
        #     payload_bytes=PAYLOAD_BYTES,
        #     debias_funcs=debias_strategy[strategy],
        # ))
        # tasks.append(dict(
        #     pcap_dict_path="/mnt/ssd1/wtz_nta_dataset/CrossPlatform/uni-flows-ios_sampled.json",
        #     save_dir=f"{dir_name}/CrossPlatform-iOS/{DATA_PREFIX}",
        #     byte_packet_num=BYTE_PACKET_NUM,
        #     seq_packet_num=SEQ_PACKET_NUM,
        #     multi_chunk=False,
        #     header_bytes=HEADER_BYTES,
        #     payload_bytes=PAYLOAD_BYTES,
        #     debias_funcs=debias_strategy[strategy],
        # ))
        # tasks.append(dict(
        #     pcap_dict_path="/mnt/ssd1/wtz_nta_dataset/DataCon2020-Malware/uni-flows_sampled.json",
        #     save_dir=f"{dir_name}/DataCon2020-Malware/{DATA_PREFIX}",
        #     byte_packet_num=BYTE_PACKET_NUM,
        #     seq_packet_num=SEQ_PACKET_NUM,
        #     multi_chunk=False,
        #     header_bytes=HEADER_BYTES,
        #     payload_bytes=PAYLOAD_BYTES,
        #     debias_funcs=debias_strategy[strategy],
        # ))
        # tasks.append(dict(
        #     pcap_dict_path="/mnt/ssd1/wtz_nta_dataset/DataCon2021-Proxy/datacon2021_eta-part1-uni-flows_sampled.json",
        #     save_dir=f"{dir_name}/DataCon2021-p1/{DATA_PREFIX}",
        #     byte_packet_num=BYTE_PACKET_NUM,
        #     seq_packet_num=SEQ_PACKET_NUM,
        #     multi_chunk=False,
        #     header_bytes=HEADER_BYTES,
        #     payload_bytes=PAYLOAD_BYTES,
        #     debias_funcs=debias_strategy[strategy],
        # ))
        # tasks.append(dict(
        #     pcap_dict_path="/mnt/ssd1/wtz_nta_dataset/DataCon2021-Proxy/datacon2021_eta-part2-uni-flows_sampled.json",
        #     save_dir=f"{dir_name}/DataCon2021-p2/{DATA_PREFIX}",
        #     byte_packet_num=BYTE_PACKET_NUM,
        #     seq_packet_num=SEQ_PACKET_NUM,
        #     multi_chunk=False,
        #     header_bytes=HEADER_BYTES,
        #     payload_bytes=PAYLOAD_BYTES,
        #     debias_funcs=debias_strategy[strategy],
        # ))
        # tasks.append(dict(
        #     pcap_dict_path="/mnt/ssd1/wtz_nta_dataset/CICIoT2022/uni-flows_ratio_sampled.json",
        #     save_dir=f"{dir_name}/CICIoT2022-ratio/{DATA_PREFIX}",
        #     byte_packet_num=BYTE_PACKET_NUM,
        #     seq_packet_num=SEQ_PACKET_NUM,
        #     multi_chunk=False,
        #     header_bytes=HEADER_BYTES,
        #     payload_bytes=PAYLOAD_BYTES,
        #     debias_funcs=debias_strategy[strategy],
        # ))
        # tasks.append(dict(
        #     pcap_dict_path="/mnt/ssd1/wtz_nta_dataset/ISCXVPN2016/uni-flows_ratio_sampled.json",
        #     save_dir=f"{dir_name}/ISCXVPN2016-ratio/{DATA_PREFIX}",
        #     byte_packet_num=BYTE_PACKET_NUM,
        #     seq_packet_num=SEQ_PACKET_NUM,
        #     multi_chunk=False,
        #     header_bytes=HEADER_BYTES,
        #     payload_bytes=PAYLOAD_BYTES,
        #     debias_funcs=debias_strategy[strategy],
        # ))
        # tasks.append(dict(
        #     pcap_dict_path="/mnt/ssd1/wtz_nta_dataset/USTC-TFC2016/uni-flows_ratio_sampled.json",
        #     save_dir=f"{dir_name}/USTC-TFC2016-ratio/{DATA_PREFIX}",
        #     byte_packet_num=BYTE_PACKET_NUM,
        #     seq_packet_num=SEQ_PACKET_NUM,
        #     multi_chunk=False,
        #     header_bytes=HEADER_BYTES,
        #     payload_bytes=PAYLOAD_BYTES,
        #     debias_funcs=debias_strategy[strategy],
        # ))
        # tasks.append(dict(
        #     pcap_dict_path="/mnt/ssd1/wtz_nta_dataset/CICIoT2022/uni-flows_m6000_sampled.json",
        #     save_dir=f"{dir_name}/CICIoT2022-m6000/{DATA_PREFIX}",
        #     byte_packet_num=BYTE_PACKET_NUM,
        #     seq_packet_num=SEQ_PACKET_NUM,
        #     multi_chunk=False,
        #     header_bytes=HEADER_BYTES,
        #     payload_bytes=PAYLOAD_BYTES,
        #     debias_funcs=debias_strategy[strategy],
        # ))
        # tasks.append(dict(
        #     pcap_dict_path="/mnt/ssd1/wtz_nta_dataset/ISCXVPN2016/uni-flows_m4000_sampled.json",
        #     save_dir=f"{dir_name}/ISCXVPN2016-m4000/{DATA_PREFIX}",
        #     byte_packet_num=BYTE_PACKET_NUM,
        #     seq_packet_num=SEQ_PACKET_NUM,
        #     multi_chunk=False,
        #     header_bytes=HEADER_BYTES,
        #     payload_bytes=PAYLOAD_BYTES,
        #     debias_funcs=debias_strategy[strategy],
        # ))

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(pcap_to_ft_json, **task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in task: {e}")

if __name__ == "__main__":
    build_pt_data()
    # build_ft_data()
