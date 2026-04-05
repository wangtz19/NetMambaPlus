import os
from scapy.all import rdpcap
import shutil
from tqdm import tqdm
import json
import concurrent.futures


def check_dns(packet):
    if packet.haslayer("DNS"):
        return True
    if packet.haslayer("UDP") and (packet["UDP"].dport == 53 or packet["UDP"].sport == 53):
        return True
    return False


def check_mdns(packet):
    if packet.haslayer("MDNS"):
        return True
    if packet.haslayer("UDP") and (packet["UDP"].dport == 5353 or packet["UDP"].sport == 5353):
        return True
    return False


def check_llmnr(packet):
    if packet.haslayer("LLMNR"):
        return True
    if packet.haslayer("UDP") and (packet["UDP"].dport == 5355 or packet["UDP"].sport == 5355):
        return True
    return False


def check_nbns(packet):
    if packet.haslayer("NBNS"):
        return True
    if packet.haslayer("UDP") and (packet["UDP"].dport == 137 or packet["UDP"].sport == 137):
        return True
    return False


def check_dhcp(packet):
    if packet.haslayer("DHCP"):
        return True
    if packet.haslayer("UDP") and (packet["UDP"].dport in [67, 68] or packet["UDP"].sport in [67, 68]):
        return True
    return False


def check_ntp(packet):
    if packet.haslayer("NTP"):
        return True
    if packet.haslayer("UDP") and (packet["UDP"].dport == 123 or packet["UDP"].sport == 123):
        return True
    return False


func_map = {
    "DNS": check_dns,
    "MDNS": check_mdns,
    "LLMNR": check_llmnr,
    "NBNS": check_nbns,
    "DHCP": check_dhcp,
    "NTP": check_ntp,
}


def clean_single_pcap(input_pcap_path: str, output_pcap_path: str,
                      removed_l7_protocols: list):
    basename = os.path.basename(input_pcap_path)
    if "TCP" not in basename and "UDP" not in basename:
        print(f"Skipping non-IP pcap: {input_pcap_path}")
        return False
    pkts = rdpcap(input_pcap_path)
    for proto in removed_l7_protocols:
        proto_checker = func_map.get(proto, lambda x: False)
        for pkt in pkts:
            if proto_checker(pkt):
                print(f"Skipping {proto} pcap: {input_pcap_path}")
                return False
    shutil.copyfile(input_pcap_path, output_pcap_path)
    return True


def clean_pcap_dataset(root_dir: str, in_sub_dir: str="uni-flows",
                       out_sub_dir: str="uni-flows-cleaned",
                       removed_l7_protocols: list=["DNS", "MDNS", "LLMNR", "NBNS","DHCP", "NTP"]):
    raw_dir = os.path.join(root_dir, in_sub_dir)
    clean_dir = os.path.join(root_dir, out_sub_dir)
    if os.path.exists(clean_dir):
        shutil.rmtree(clean_dir)
    os.makedirs(clean_dir, exist_ok=True)
    stats = {}
    for app_name in os.listdir(raw_dir):
        app_raw_dir = os.path.join(raw_dir, app_name)
        app_clean_dir = os.path.join(clean_dir, app_name)
        os.makedirs(app_clean_dir, exist_ok=True)
        pcap_files = [f for f in os.listdir(app_raw_dir) if f.endswith(".pcap")]
        stats[app_name] = {"total": len(pcap_files), "cleaned": 0}
        for pcap_file in tqdm(pcap_files, desc=f"Cleaning {app_name}"):
            input_pcap_path = os.path.join(app_raw_dir, pcap_file)
            output_pcap_path = os.path.join(app_clean_dir, pcap_file)
            if clean_single_pcap(input_pcap_path, output_pcap_path, removed_l7_protocols):
                stats[app_name]["cleaned"] += 1
    with open(os.path.join(root_dir, "cleaning_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


def clean_all(max_workers=4):
    tasks = []
    for root_dir in [
        "/mnt/ssd1/wtz_nta_dataset/ISCXVPN2016",
        "/mnt/ssd1/wtz_nta_dataset/USTC-TFC2016",
        "/mnt/ssd1/wtz_nta_dataset/CICIoT2022",
    ]:
        tasks.append(concurrent.futures.ProcessPoolExecutor().submit(clean_pcap_dataset,
            root_dir=root_dir,
            in_sub_dir="uni-flows",
            out_sub_dir="uni-flows-cleaned",
        ))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(clean_pcap_dataset, **task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in task: {e}")

if __name__ == "__main__":
    clean_all()
