import os
from tqdm import tqdm
import subprocess
import sys
sys.path.append("/root/Vim/dataset_scripts")
from dataset_common import find_files
import shutil

dir_name = "/mnt/ssd1/wtz_nta_dataset/CICIoT2022"
splitter = "/mnt/ssd2/wtz/ShieldGPT/pcap_tool/SplitCap.exe"

def split_pcap_files(granularity="session"):
    if granularity == "session":
        flow_name = "bi-flows"
    elif granularity == "flow":
        flow_name = "uni-flows"
    else:
        raise ValueError("Granularity must be either 'session' or 'flow'.")
    categories = ["1-Power", "6-Attacks"]
    for category in categories:
        sub_categories = list(filter(lambda x: os.path.isdir(os.path.join(dir_name, category, x)), 
                            os.listdir(os.path.join(dir_name, category))))
        for sub_category in sub_categories:
            out_dir = f"{dir_name}/{flow_name}/{category}-{sub_category}" # merge sub-category with category
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)
            log_dir = f"{dir_name}/log/{flow_name}/{category}-{sub_category}"
            os.makedirs(log_dir, exist_ok=True)
            pcapfiles = find_files(os.path.join(dir_name, category, sub_category), ".pcap")
            for filename in tqdm(pcapfiles, desc=f"Splitting {category}/{sub_category}"):
                with open(f"{log_dir}/{os.path.basename(filename)}.log", "w") as f:
                    subprocess.run(f"mono {splitter} -r '{filename}' -o '{out_dir}' -s {granularity}",
                                   shell=True, stdout=f, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    granularity = "session"
    split_pcap_files(granularity)
