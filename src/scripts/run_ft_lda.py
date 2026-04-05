import os
import subprocess
import json

BYTE_PACKET_NUM = 5
SEQ_PACKET_NUM = 50
STRIDE_LEN = 4
HEADER_BYTE_NUM = 80
PAYLOAD_BYTE_NUM = 240

def run(cuda_idx, dataset, model_name, ckpt_step=100000, epochs=120, strategy="none"):
    ckpt_path = "" # TODO
    data_dir = "" # TODO
    output_dir = "" # TODO
    with open(f"{data_dir}/metadata.json", "r") as f:
        metadata = json.load(f)
    num_class = len(metadata["name_to_idx"])
    os.makedirs(output_dir, exist_ok=True)
    cmd = f"""
CUDA_VISIBLE_DEVICES={cuda_idx} python -u ../fine-tune.py \\
    --num_packet {BYTE_PACKET_NUM} \\
    --num_packet_byte {HEADER_BYTE_NUM + PAYLOAD_BYTE_NUM} \\
    --stride_size {STRIDE_LEN} \\
    --batch_size 128 \\
    --epochs {epochs} \\
    --nb_classes {num_class} \\
    --finetune "{ckpt_path}" \\
    --data_path {data_dir} \\
    --output_dir "{output_dir}" \\
    --log_dir "{output_dir}" \\
    --model net_{model_name}_classifier \\
    --no_amp \\
    --ldam \\
    --class_balance \\
    > "{output_dir}/finetune.txt" 2>&1 &
"""
    print(cmd)
    subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    for model_name in ["fgt_base", "mamba"]: # ["fgt_base", "mamba"]:
        for strategy in ["zero_ip_port"]:
            run(cuda_idx=3, dataset="CICIoT2022", model_name=model_name, ckpt_step=100000, strategy=strategy)
            run(cuda_idx=2, dataset="CipherSpectrum", model_name=model_name, ckpt_step=100000, strategy=strategy)
            run(cuda_idx=1, dataset=["CrossNet2021", "ScenarioA"], model_name=model_name, ckpt_step=100000, strategy=strategy)
            run(cuda_idx=0, dataset=["CrossNet2021", "ScenarioB"], model_name=model_name, ckpt_step=100000, strategy=strategy)
            run(cuda_idx=3, dataset="CSTNET-TLS1.3", model_name=model_name, ckpt_step=100000, strategy=strategy)
            run(cuda_idx=2, dataset="ISCXVPN2016", model_name=model_name, ckpt_step=100000, strategy=strategy)
            run(cuda_idx=1, dataset="USTC-TFC2016", model_name=model_name, ckpt_step=100000, strategy=strategy)
            # run(cuda_idx=0, dataset="DataCon2020-Malware", model_name=model_name, ckpt_step=100000, strategy=strategy)
            # run(cuda_idx=3, dataset="DataCon2021-p1", model_name=model_name, ckpt_step=100000, strategy=strategy)
            # run(cuda_idx=3, dataset="DataCon2021-p2", model_name=model_name, ckpt_step=100000, strategy=strategy)
            run(cuda_idx=0, dataset="CrossPlatform-Android", model_name=model_name, ckpt_step=100000, strategy=strategy)
            run(cuda_idx=3, dataset="CrossPlatform-iOS", model_name=model_name, ckpt_step=100000, strategy=strategy)
            # run(cuda_idx=1, dataset="CICIoT2022-ratio", model_name=model_name, ckpt_step=100000, strategy=strategy)
            # run(cuda_idx=2, dataset="ISCXVPN2016-ratio", model_name=model_name, ckpt_step=100000, strategy=strategy)
            # run(cuda_idx=3, dataset="USTC-TFC2016-ratio", model_name=model_name, ckpt_step=100000, strategy=strategy)
            # run(cuda_idx=0, dataset="CICIoT2022-m6000", model_name=model_name, ckpt_step=100000, strategy=strategy)
            # run(cuda_idx=1, dataset="ISCXVPN2016-m4000", model_name=model_name, ckpt_step=100000, strategy=strategy)
