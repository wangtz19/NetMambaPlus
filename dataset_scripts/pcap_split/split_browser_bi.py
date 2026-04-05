import os
from tqdm import tqdm
import subprocess
import shutil

dir_name = "/mnt/ssd1/wtz_nta_dataset/Browser"
splitter = "/mnt/ssd2/wtz/ShieldGPT/pcap_tool/SplitCap.exe"

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
    granularity = "flow"
    split_pcap_files(granularity)
