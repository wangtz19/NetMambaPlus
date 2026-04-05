import os
from tqdm import tqdm
import subprocess
import sys
sys.path.append("/root/Vim/dataset_scripts")
from dataset_common import find_files
import shutil


device_prefix_dict = {
    "Audio": ["Amazon Echo Dot", "Amazon Echo Spot", "Amazon Echo Studio", "Google Nest Mini", "Sonos One Speaker"], # merge "Amazon Echo Dot *"
    "Cameras": ["Amcrest", "Arlo Basestation Camera", "ArloQ Camera", "Borun Camera", "DLink Camera", "HeimVision Camera",
                "Home Eye Camera", "Luohe Camera", "Nest Camera", "Netatmo Camera", "SimCam"],
    "Home Automation": ["Amazon Plug", "Atomi Coffee Maker", "Eufy Homebase", "Globe Lamp", "Gosund Plug", "HeimVision Lamp",
                        "Philips Hue Bridge", "Ring Basestation", "Roomba Vacuum", "Smart Board", "Tekin Plug", "Yutron Plug"], # merge "Gosund Plug *", "Tekin Plug *", "Yutron Plug *"
    "Other": ["DLink Water Sensor", "LG TV"]
}

splitter = "/mnt/ssd2/wtz/ShieldGPT/pcap_tool/SplitCap.exe"

def split_pcap_files(input_dir, output_dir, prefix_dict=None, granularity="session"):
    categories = list(filter(lambda x: os.path.isdir(os.path.join(input_dir, x)), os.listdir(input_dir)))
    for category in categories:
        sub_categories = list(filter(lambda x: os.path.isdir(os.path.join(input_dir, category, x)), os.listdir(os.path.join(input_dir, category))))
        merged_sub_categories = []
        if prefix_dict is not None:
            for prefix in device_prefix_dict[category]:
                merged_sub_categories.append([prefix, list(filter(lambda x: x.startswith(prefix), sub_categories))])
        else:
            merged_sub_categories = [[x, [x]] for x in sub_categories]
        for prefix, sub_category in merged_sub_categories:
            out_dir = os.path.join(output_dir, category, prefix)
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)
            for app_name in sub_category:
                app_files = find_files(os.path.join(input_dir, category, app_name), ".pcap")
                for filename in tqdm(app_files, desc=f"Splitting {category}/{app_name}"):
                    subprocess.run(f"mono {splitter} -r '{filename}' -o '{out_dir}' -s {granularity}",
                                    shell=True, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    granularity = "flow"
    if granularity == "session":
        flow_name = "bi-flows"
    elif granularity == "flow":
        flow_name = "uni-flows"
    else:
        raise ValueError("Granularity must be either 'session' or 'flow'.")
    split_pcap_files("/mnt/ssd1/wtz_nta_dataset/ciciot2022-raw/1-Power",
                     f"/mnt/ssd1/wtz_nta_dataset/ciciot2022-raw/{flow_name}/1-Power",
                     prefix_dict=device_prefix_dict)
    split_pcap_files("/mnt/ssd1/wtz_nta_dataset/ciciot2022-raw/3-Interactions",
                     f"/mnt/ssd1/wtz_nta_dataset/ciciot2022-raw/{flow_name}/3-Interactions",
                     prefix_dict=device_prefix_dict)
    split_pcap_files("/mnt/ssd1/wtz_nta_dataset/ciciot2022-raw/6-Attacks",
                     f"/mnt/ssd1/wtz_nta_dataset/ciciot2022-raw/{flow_name}/6-Attacks")
