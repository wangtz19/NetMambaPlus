import os
import sys
sys.path.append("/root/Vim/dataset_scripts")
import concurrent.futures

from dataset_debias_common import (
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
from dataset_mm_bi_common import pcap_to_pt_json, pcap_to_ft_json


debias_strategy = {
    # "none": None,
    # "rz-ip": [relative_zero_ip],
    # "zero_ip_port_rzshortcut": [zero_ip_port, relative_zero_seq_ack_no, relative_zero_tcp_ts_option],
    "zero_ip_port": [zero_ip_port],
}


BYTE_PACKET_NUM = 10
SEQ_PACKET_NUM = 50
HEADER_BYTES = 80
PAYLOAD_BYTES = 80
DATA_DIR_NAME = f"bi_nb={BYTE_PACKET_NUM}_ns={SEQ_PACKET_NUM}_hb={HEADER_BYTES}_pb={PAYLOAD_BYTES}"

dir_name = "/mnt/ssd1/wtz_nta_dataset/netx_data/dataset_mm_bi"


def build_pt_data():
    for strategy in debias_strategy:
        DATA_PREFIX = f"{DATA_DIR_NAME}_debias={strategy}"
        # pcap_to_pt_json(
        #     pcap_dirs=["/mnt/ssd1/wtz_nta_dataset/Browser/bi-flows",
        #                "/mnt/ssd1/wtz_nids_dataset/kitsune/bi-flows",
        #                "/mnt/ssd1/wtz_nta_dataset/VNAT/bi-flows"],
        #     save_dir=f"{dir_name}/pretrain-3/{DATA_PREFIX}",
        #     filter_func=None,
        #     byte_packet_num=BYTE_PACKET_NUM,
        #     seq_packet_num=SEQ_PACKET_NUM,
        #     multi_chunk=False,
        #     header_bytes=HEADER_BYTES,
        #     payload_bytes=PAYLOAD_BYTES,
        #     debias_funcs=debias_strategy[strategy],
        # )
        pcap_to_pt_json(
            pcap_dirs=["/mnt/ssd1/wtz_nta_dataset/Browser/bi-flows",
                       "/mnt/ssd1/wtz_nids_dataset/kitsune/bi-flows",],
            save_dir=f"{dir_name}/pretrain-2/{DATA_PREFIX}",
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
        tasks.append(dict(
            pcap_dict_path="/root/garmr-data/pcap_splits.json",
            save_dir=f"/root/garmr-data/json/attack_all_train_test",
            byte_packet_num=BYTE_PACKET_NUM,
            seq_packet_num=SEQ_PACKET_NUM,
            multi_chunk=False,
            header_bytes=HEADER_BYTES,
            payload_bytes=PAYLOAD_BYTES,
            debias_funcs=debias_strategy[strategy],
        ))

        for attack in ["ids2018_bruteforce_ssh","ids2018_bruteforce_web", "ids2018_bruteforce_xss",
                       "ids2018_dos_goldeneye", "ids2018_dos_hulk", "ids2018_dos_slowloris",
                       "ton_iot_injection", "ton_iot_password", "ton_iot_xss",]:
            tasks.append(dict(
                pcap_dict_path=f"/root/garmr-data/pcap_splits_test_{attack}.json",
                save_dir=f"/root/garmr-data/json/attack_{attack}_test",
                byte_packet_num=BYTE_PACKET_NUM,
                seq_packet_num=SEQ_PACKET_NUM,
                multi_chunk=False,
                header_bytes=HEADER_BYTES,
                payload_bytes=PAYLOAD_BYTES,
                debias_funcs=debias_strategy[strategy],
            ))

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(pcap_to_ft_json, **task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in task: {e}")

if __name__ == "__main__":
    # build_pt_data()
    build_ft_data()
