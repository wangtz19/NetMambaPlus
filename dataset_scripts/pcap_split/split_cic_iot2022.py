import os
from tqdm import tqdm
import subprocess
from dataset_common import find_files

def split_pcap_files():
    filenames = find_files("/mnt/ssd1/ciciot2022-raw")
    for filename in tqdm(filenames):
        splitter = "/mnt/ssd2/ShieldGPT/pcap_tool/splitter"
        dir_name = os.path.join("/root/Vim/dataset/CICIoT2022", "flows", *filename.split("/")[4:-1])
        os.makedirs(dir_name, exist_ok=True)
        flow_prefix = os.path.basename(filename)[:-len(".pcap")]
        with open(os.path.join(dir_name, f"{flow_prefix}.log"), "w") as f:
            subprocess.run(f"{splitter} -i '{filename}' -o '{dir_name}' -p {flow_prefix}- -f five_tuple",
                        shell=True, stdout=f, stderr=subprocess.STDOUT)

# def shuffle_packets(flow_array, prob=0.15):
#     if np.random.rand() > prob:
#         data = flow_array
#         label = [1] * 5
#     else:
#         # cut the array into 5 packets with 320 integers each
#         packets = np.split(flow_array, 5)
#         # shuffle the packets and record which packets are shuffled
#         random_indices = np.random.permutation(len(packets))
#         data = np.concatenate([packets[i] for i in random_indices])
#         label = [int(x == y) for x, y in zip(range(len(packets)), random_indices)]
#     return data, label

# from concurrent.futures import ThreadPoolExecutor

# def process_file(filename, label, if_stat=False):
#     image_filename = f"CICIoT2022/images/{label}/{'-'.join(filename.split('/')[2:])}.png"
#     if os.path.exists(image_filename) and not if_stat:
#         return
#     stat_filename = image_filename.replace(".png", ".json")
#     if os.path.exists(image_filename) and os.path.exists(stat_filename):
#         return
#     try:
#         res = read_5hp_list(filename, if_stat=if_stat)[0]
#         flow_array = res.pop("data")
#         if not os.path.exists(image_filename):
#             image = Image.fromarray(flow_array.reshape((40, 40)).astype(np.uint8))
#             image.save(image_filename)
#         if if_stat and not os.path.exists(stat_filename):
#             with open(stat_filename, "w") as f:
#                 json.dump(res, f)
#     except Exception as e:
#         print(f"Error processing {filename}: {e}")

# def pcap_to_image(if_stat=False):
#     label_dict = {
#         "Power-Audio": "1-Power/Audio",
#         "Power-Cameras": "1-Power/Cameras",
#         "Power-HomeAutomation": "1-Power/Home Automation",
#         "Attacks-Flood": "6-Attacks/1-Flood",
#         "Attacks-Hydra": "6-Attacks/2-RTSP Brute Force/Hydra",
#         "Attacks-Nmap": "6-Attacks/2-RTSP Brute Force/Nmap",
#     }

#     with ThreadPoolExecutor(max_workers=10) as executor:
#         for label, label_dir in label_dict.items():
#             filenames = find_files(f"CICIoT2022/flows/{label_dir}", ".pcap")
#             os.makedirs(f"CICIoT2022/images/{label}", exist_ok=True)
#             for filename in tqdm(filenames):
#                 executor.submit(process_file, filename, label, if_stat)

if __name__ == "__main__":
    split_pcap_files()