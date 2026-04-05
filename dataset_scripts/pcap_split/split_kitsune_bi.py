import shutil
import os
import subprocess
from tqdm import tqdm


attacks = ["Active Wiretap", "ARP MitM", "Fuzzing", "Mirai Botnet", "OS Scan",
           "SSDP Flood", "SSL Renegotiation", "SYN DoS", "Video Injection"]
dir_name = "/mnt/ssd1/wtz_nids_dataset/kitsune"
splitter = "/mnt/ssd2/wtz/ShieldGPT/pcap_tool/SplitCap.exe"

def prepare_pcaps():
    os.makedirs(f"{dir_name}/pcap", exist_ok=True)
    for attack in attacks:
        pcap_files = list(filter(lambda x: x.endswith(".pcap"),
                                os.listdir(f"{dir_name}/raw/{attack}")))
        pcapng_files = list(filter(lambda x: x.endswith(".pcapng"),
                                os.listdir(f"{dir_name}/raw/{attack}")))
        print(f"{attack}: {pcap_files}, {pcapng_files}")
        for pcap_file in pcap_files:
            # some files are ended with .pcap but are actually pcapng files
            cmd = f"editcap -F pcap '{dir_name}/raw/{attack}/{pcap_file}' {dir_name}/pcap/{pcap_file[:-len('.pcap')]}.pcap"
            subprocess.run(cmd, shell=True)
        for pcapng_file in pcapng_files:
            cmd = f"editcap -F pcap '{dir_name}/raw/{attack}/{pcapng_file}' {dir_name}/pcap/{pcapng_file[:-len('.pcapng')]}.pcap"
            subprocess.run(cmd, shell=True)

def split_pcap_files(granularity="session"):
    pcap_filenames = list(filter(lambda x: x.endswith(".pcap"), 
                        os.listdir(f"{dir_name}/pcap")))
    if granularity == "session":
        flow_name = "bi-flows"
    elif granularity == "flow":
        flow_name = "uni-flows"
    else:
        raise ValueError("Granularity must be either 'session' or 'flow'.")
    for filename in tqdm(pcap_filenames, desc=f"Splitting"):
        app_name = filename[:-len('.pcap')]
        out_dir = f"{dir_name}/{flow_name}/{app_name}"
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        log_dir = f"{dir_name}/log/{flow_name}"
        os.makedirs(log_dir, exist_ok=True)
        with open(f"{log_dir}/{app_name}.log", "w") as f:
            subprocess.run(f"mono {splitter} -r '{dir_name}/pcap/{filename}' -o '{out_dir}' -s {granularity}",
                        shell=True, stdout=f, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    granularity = "session"
    # prepare_pcaps()
    split_pcap_files(granularity)
    print("Splitting completed.")