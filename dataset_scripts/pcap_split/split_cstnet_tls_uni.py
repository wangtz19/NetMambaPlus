import os
from tqdm import tqdm
import subprocess
import shutil

splitter = "/mnt/ssd2/wtz/ShieldGPT/pcap_tool/SplitCap.exe"

def split_pcap_files(in_dir, out_dir, log_dir):
    app_names = list(filter(lambda x: os.path.isdir(f"{in_dir}/{x}"), 
                        os.listdir(in_dir)))
    for app_name in tqdm(app_names, desc=f"Splitting"):
        filenames = list(filter(lambda x: x.endswith(".pcap"), 
                                os.listdir(f"{in_dir}/{app_name}")))
        if os.path.exists(f"{out_dir}/{app_name}"):
            shutil.rmtree(f"{out_dir}/{app_name}")
        os.makedirs(f"{out_dir}/{app_name}", exist_ok=True)
        os.makedirs(f"{log_dir}/{app_name}", exist_ok=True)
        for filename in filenames:
            with open(f"{log_dir}/{app_name}/{filename}.log", "w") as f:
                subprocess.run(f"mono {splitter} -r '{in_dir}/{app_name}/{filename}' -o '{out_dir}/{app_name}' -s flow",
                            shell=True, stdout=f, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    # split_pcap_files("/mnt/ssd1/wtz_nta_dataset/CSTNET-TLS1.3/bi-flows",
    #                  "/mnt/ssd1/wtz_nta_dataset/CSTNET-TLS1.3/uni-flows",
    #                 "/mnt/ssd1/wtz_nta_dataset/CSTNET-TLS1.3/log/uni-flows")
    # split_pcap_files("/mnt/ssd1/wtz_nta_dataset/CipherSpectrum/bi-flows/mix",
    #                  "/mnt/ssd1/wtz_nta_dataset/CipherSpectrum/uni-flows/mix",
    #                 "/mnt/ssd1/wtz_nta_dataset/CipherSpectrum/log/uni-flows/mix")
    # split_pcap_files("/mnt/ssd1/wtz_nta_dataset/CipherSpectrum/bi-flows/aes-256-gcm",
    #                  "/mnt/ssd1/wtz_nta_dataset/CipherSpectrum/uni-flows/aes-256-gcm",
    #                 "/mnt/ssd1/wtz_nta_dataset/CipherSpectrum/log/uni-flows/aes-256-gcm")
    # split_pcap_files("/mnt/ssd1/wtz_nta_dataset/CipherSpectrum/bi-flows/chacha20-poly1305",
    #                  "/mnt/ssd1/wtz_nta_dataset/CipherSpectrum/uni-flows/chacha20-poly1305",
    #                 "/mnt/ssd1/wtz_nta_dataset/CipherSpectrum/log/uni-flows/chacha20-poly1305")
    split_pcap_files("/mnt/ssd1/wtz_nta_dataset/CipherSpectrum/bi-flows/aes-128-gcm",
                     "/mnt/ssd1/wtz_nta_dataset/CipherSpectrum/uni-flows/aes-128-gcm",
                    "/mnt/ssd1/wtz_nta_dataset/CipherSpectrum/log/uni-flows/aes-128-gcm")
