# NetMamba+


## Envonriment Setup
- Create conda environment with python-3.10.13
  ```
  conda create -n MAMBA python=3.10.13
  conda activate MAMBA
  ```
- Install torch-2.2.0 + torchvision-0.17.0 with cu121
  ```
  pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
  ```
- Install mamba_ssm-1.1.1 + causal_conv1d-1.1.2.post1
  ```
  cd mamba-1p1p1
  pip install --no-build-isolation .
  ```
- Install flash-attn-2.7.4.post1
  ```
  pip install flash-attn==2.7.4.post1 --no-build-isolation
  ```
- Install others
  ```
  pip install -r requirements.txt
  ```


## Prepare Datasets
- (1) Split raw pcap files into flows (uni-directional or bi-directional). 
  -  Refer to `pcap_split`, you may need to write new scripts for new datasets.

- (2) Count and sample datasets if in need. 
  - Refer to `pcap_sample/sample_pcap.py` for sampling data and splitting them into `train / valid / test` sets of each dataset.

- (3) Extract flow-level byte and sequence features.
  - Refer to `feat_extract/feat_mm_bi_flows` for bi-directional flows.
  - Refer to `feat_extract/feat_mm_uni_flows` for uni-directional flows.


## Training and Evaluation
- Pre-training
```
cd src/scripts
python run_pt.py
```
You need to modify file paths accordingly.

- Fine-tuning and Evaluation
```
cd src/scripts
python run_ft.py # without lda
python run_ft_lda.py # with lda
```
You need to modify file paths accordingly.