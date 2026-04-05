import os
from tqdm import tqdm
import subprocess
import shutil


dir_name = "/mnt/ssd1/wtz_nta_dataset/CICMalAnal2017"
splitter = "/mnt/ssd2/wtz/ShieldGPT/pcap_tool/SplitCap.exe"

def split_pcap_files():
    # for mal_family in ["Adware", "Ransomware", "Scareware", "SMSMalware"]:
    for mal_family in ["Ransomware"]:
        data_dir = f"{dir_name}/{mal_family}"
        apps = list(filter(lambda x: os.path.isdir(f"{data_dir}/{x}"), os.listdir(data_dir)))
        for app in tqdm(apps):
            out_dir = f"{dir_name}/bi-flows/{mal_family}-{app}"
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)
            with open(f"{out_dir}.log", "w") as f:
                for filename in os.listdir(f"{data_dir}/{app}"):
                    subprocess.run(f"mono {splitter} -r '{data_dir}/{app}/{filename}' -o '{out_dir}' -s session",
                                shell=True, stdout=f, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    split_pcap_files()
