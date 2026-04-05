import os
import subprocess

BYTE_PACKET_NUM = 5
SEQ_PACKET_NUM = 20
STRIDE_LEN = 4
HEADER_BYTE_NUM = 80
PAYLOAD_BYTE_NUM = 240


DATA_PREFIX = f"uni_nb={BYTE_PACKET_NUM}_ns=50_hb={HEADER_BYTE_NUM}_pb={PAYLOAD_BYTE_NUM}_debias=none"
SAVE_PREFIX = f"uni_nb={BYTE_PACKET_NUM}_ns={SEQ_PACKET_NUM}_hb={HEADER_BYTE_NUM}_pb={PAYLOAD_BYTE_NUM}_debias=none_s={STRIDE_LEN}"

def run(cuda_idx, model_name, steps, byte_mask_ratio=0.9):
    output_dir = "" # TODO
    data_path = "" # TODO
    if model_name.startswith("fuse3_"):
        dataset_type = "byte_size_interval"
    else:
        dataset_type = "byte_size"
    if "iat" in model_name:
        seq_key = "intervals"
    else:
        seq_key = "sizes"
    os.makedirs(output_dir, exist_ok=True)
    cmd = f"""
CUDA_VISIBLE_DEVICES={cuda_idx} python -u ../pre-train.py \\
    --num_packet {BYTE_PACKET_NUM} \\
    --num_packet_byte {HEADER_BYTE_NUM + PAYLOAD_BYTE_NUM} \\
    --stride_size {STRIDE_LEN} \\
    --seq_len {SEQ_PACKET_NUM} \\
    --batch_size 128 \\
    --blr 1e-3 \\
    --steps {steps} \\
    --data_path {data_path} \\
    --output_dir "{output_dir}" \\
    --log_dir "{output_dir}" \\
    --model {model_name}_pretrain \\
    --dataset_type {dataset_type} \\
    --seq_key {seq_key} \\
    --byte_mask_ratio {byte_mask_ratio} \\
    --no_amp \\
    > "{output_dir}/pretrain.txt" 2>&1 &
"""
    print(cmd)
    subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    # run(cuda_idx=0, model_name="fuse3_mamba", steps=100000)
    run(cuda_idx=3, model_name="fuse3_bgt_base", steps=100000)
