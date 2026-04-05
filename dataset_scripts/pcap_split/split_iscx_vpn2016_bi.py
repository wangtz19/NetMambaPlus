import os
from tqdm import tqdm
import subprocess
import shutil

dir_name = "/mnt/ssd1/wtz_nta_dataset/ISCXVPN2016"
splitter = "/mnt/ssd2/wtz/ShieldGPT/pcap_tool/SplitCap.exe"

def rename_pcap_files():
    for cat in ["VPN", "NonVPN"]:
        filenames = os.listdir(f"{dir_name}/{cat}")
        for filename in tqdm(filenames):
            new_filename = filename.strip(" ")
            os.rename(os.path.join(f"{dir_name}/{cat}", filename), os.path.join(f"{dir_name}/{cat}", new_filename))

def pcapng_to_pcap():
    for cat in ["VPN", "NonVPN"]:
        pcapng_files = list(filter(lambda x: x.endswith("pcapng"), 
                                   os.listdir(f"{dir_name}/{cat}")))
        for pcapng_file in tqdm(pcapng_files):
            pcapng_file = os.path.join(f"{dir_name}/{cat}", pcapng_file)
            cmd = f"editcap {pcapng_file} {pcapng_file[:-len('.pcapng')] + '.pcap'}"
            print(cmd)
            subprocess.run(cmd, shell=True)
            os.remove(pcapng_file)

category2keywords = {
    "VPN": {
        "browsing": ["vpn_netflix", "vpn_spotify", "vpn_voipbuster"],
        "email": ["vpn_email"],
        "chat": ["vpn_icq_chat", "vpn_aim_chat", "vpn_skype_chat", "vpn_facebook_chat", "vpn_hangouts_chat"],
        "streaming": ["vpn_vimeo", "vpn_youtube"],
        "file": ["vpn_ftp", "vpn_sftp", "vpn_skype_files"],
        "voip": ["vpn_skype_audio", "vpn_facebook_audio", "vpn_hangouts_audio"],
        "p2p": ["vpn_bittorrent"]
    },
    "NonVPN": {
        "browsing": ["netflix", "spotify", "voipbuster"],
        "email": ["email"],
        "chat": ["icq_chat", "ICQchat", "aim_chat", "AIMchat", "skype_chat", "facebook_chat", "hangouts_chat", "gmailchat",
                 "facebookchat", ],
        "streaming": ["vimeo", "youtube"],
        "file": ["ftps", "sftp", "skype_file", "scp"],
        "voip": ["skype_audio", "facebook_audio", "hangouts_audio"],
    }
}

def sanity_check():
    for cat in ["VPN", "NonVPN"]:
        remaining_files = set(os.listdir(f"{dir_name}/{cat}"))
        n_files = len(remaining_files) 
        for label, keywords in category2keywords[cat].items():
            filenames = []
            for keyword in keywords:
                tmp = list(filter(lambda x: x.endswith(".pcap") and x.startswith(keyword), 
                                os.listdir(f"{dir_name}/{cat}")))
                assert len(tmp) > 0, f"Error: {keyword} not found in {dir_name}/{cat} for {label}"
                filenames += tmp
            for filename in filenames:
                remaining_files.remove(filename)
        n_remaining_files = len(remaining_files)
        print(f"Category: {cat}, Remaining ratio={n_remaining_files/n_files:.4f}, Remaining files: {remaining_files}")

def split_pcap_files(granularity="session"):
    if granularity == "session":
        flow_name = "bi-flows"
    elif granularity == "flow":
        flow_name = "uni-flows"
    else:
        raise ValueError("Granularity must be either 'session' or 'flow'.")
    for cat in ["VPN", "NonVPN"]:
        for label, keywords in category2keywords[cat].items():
            filenames = []
            for keyword in keywords:
                filenames += list(filter(lambda x: x.endswith(".pcap") and x.startswith(keyword), 
                                os.listdir(f"{dir_name}/{cat}")))
            filenames = [f"{dir_name}/{cat}/{x}" for x in filenames]
            out_dir = f"{dir_name}/{flow_name}/{cat}-{label}"
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)
            log_dir = f"{dir_name}/log/{flow_name}/{cat}-{label}"
            os.makedirs(log_dir, exist_ok=True)
            for filename in tqdm(filenames, desc=f"Processing {cat}/{label}"):
                with open(f"{log_dir}/{os.path.basename(filename)[:-len('.pcap')]}.log", "w") as f:
                    subprocess.run(f"mono {splitter} -r '{filename}' -o '{out_dir}' -s {granularity}",
                                shell=True, stdout=f, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    granularity = "session"
    # rename_pcap_files()
    # pcapng_to_pcap()
    sanity_check()
    split_pcap_files(granularity)