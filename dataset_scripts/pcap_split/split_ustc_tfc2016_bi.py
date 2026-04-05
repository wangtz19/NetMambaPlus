import os
from tqdm import tqdm
import subprocess
import shutil

dir_name = "/mnt/ssd1/wtz_nta_dataset/USTC-TFC2016"
splitter = "/mnt/ssd2/wtz/ShieldGPT/pcap_tool/SplitCap.exe"

def split_pcap_files(granularity="session"):
    if granularity == "session":
        flow_name = "bi-flows"
    elif granularity == "flow":
        flow_name = "uni-flows"
    else:
        raise ValueError("Granularity must be either 'session' or 'flow'.")
    log_dir = f"{dir_name}/log/{flow_name}"
    os.makedirs(log_dir, exist_ok=True)
    for label in ["Benign", "Malware"]:
        pcap_filenames = list(filter(lambda x: x.endswith(".pcap") or os.path.isdir(x), 
                            os.listdir(f"{dir_name}/{label}")))
        for filename in tqdm(pcap_filenames, desc=f"Splitting {label}"):
            app_name = filename[:-len('.pcap')]
            out_dir = f"{dir_name}/{flow_name}/{label}-{app_name}"
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)
            with open(f"{log_dir}/{label}-{app_name}.log", "w") as f:
                subprocess.run(f"mono {splitter} -r '{dir_name}/{label}/{filename}' -o '{out_dir}' -s {granularity}",
                            shell=True, stdout=f, stderr=subprocess.STDOUT)
    for app_name in ["SMB", "Weibo"]:
        filenames = os.listdir(f"{dir_name}/Benign/{app_name}")
        out_dir = f"{dir_name}/{flow_name}/Benign-{app_name}"
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        for idx, filename in enumerate(filenames):
            with open(f"{log_dir}/{app_name}-{idx}.log", "w") as f:
                subprocess.run(f"mono {splitter} -r '{dir_name}/Benign/{app_name}/{filename}' -o '{out_dir}' -s {granularity}",
                            shell=True, stdout=f, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    granularity = "flow"
    split_pcap_files(granularity)