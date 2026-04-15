# Dataset Scripts

Scripts for processing raw PCAP network traffic captures into structured JSON datasets suitable for training and evaluating NetMamba+. The pipeline covers the full lifecycle: splitting raw PCAPs into individual flows, cleaning out non-application-layer protocols, sampling and partitioning into train/valid/test splits, extracting multi-modal features (packet bytes + sequence-level statistics), and inspecting dataset properties.

## Directory Structure

```
dataset_scripts/
├── dataset_json_common.py        # Shared packet-level anonymization and debiasing functions
├── dataset_mm_bi_common.py       # Bidirectional flow feature extraction utilities
├── dataset_mm_uni_common.py      # Unidirectional flow feature extraction utilities
├── pcap_split/                   # Stage 1: Split raw PCAPs into per-flow PCAPs
│   ├── split_browser_bi.py           # Browser dataset
│   ├── split_cic_andmal_bi.py        # CIC Android Malware 2017
│   ├── split_cic_iot2022.py          # CIC IoT 2022 (five-tuple splitter)
│   ├── split_cic_iot2022_bi.py       # CIC IoT 2022 (SplitCap, bidirectional)
│   ├── split_cic_iot2022_uni.py      # CIC IoT 2022 (SplitCap, unidirectional)
│   ├── split_crossnet_bi.py          # CrossNet 2021 (Scenario A & B)
│   ├── split_cstnet_tls_uni.py       # CSTNET-TLS 1.3 / CipherSpectrum
│   ├── split_iscx_vpn2016.py         # ISCX VPN 2016 (five-tuple splitter)
│   ├── split_iscx_vpn2016_bi.py      # ISCX VPN 2016 (SplitCap)
│   ├── split_kitsune_bi.py           # Kitsune NIDS
│   ├── split_ustc_tfc2016.py         # USTC-TFC 2016 (five-tuple splitter)
│   └── split_ustc_tfc2016_bi.py      # USTC-TFC 2016 (SplitCap)
├── pcap_clean/                   # Stage 2: Remove non-application-layer flows
│   └── clean_pcap.py
├── pcap_sample/                  # Stage 3: Sample flows and create train/valid/test splits
│   └── sample_pcap.py
├── feat_extract/                 # Stage 4: Extract multi-modal features into JSON
│   ├── feat_mm_bi_flows.py           # Bidirectional flow feature extraction entry point
│   └── feat_mm_uni_flows.py          # Unidirectional flow feature extraction entry point
└── data_inspect/                 # Auxiliary: Dataset analysis and visualization
    ├── analyse_byte_ami.py           # Adjusted Mutual Information of raw byte ranges
    ├── analyse_seq_ami.py            # AMI of sequence-level features (sizes, intervals)
    ├── analyse_flow_duration.py      # Flow duration timeline visualization
    ├── analyse_pcap_tls.py           # TLS/SSL/QUIC protocol and cipher statistics
    ├── run_analyse_pcap_tls.sh       # Batch runner for TLS analysis
    └── id-cipher.csv                 # TLS cipher suite ID-to-name mapping table
```

## Root-Level Common Modules

### `dataset_json_common.py` — Packet Anonymization & Debiasing

A library of functions that transform packet fields in a `scapy.PacketList` to remove or randomize information that could introduce shortcut bias in classification models. Every function takes a `PacketList` and returns a modified `PacketList`. They are composed into "debias strategies" by the feature extraction scripts.

| Category | Functions | Description |
|---|---|---|
| **IP address** | `zero_ip`, `zero_ip_port`, `relative_zero_ip`, `random_ip_port` | Zero out, relativize (first-src → `0.0.0.0`, other → `0.0.0.1`), or randomize IP addresses. Supports both IPv4 and IPv6. |
| **Port** | `zero_port`, `relative_zero_port` | Zero out or relativize transport-layer ports. |
| **TCP seq/ack** | `zero_seq_ack_no`, `relative_zero_seq_ack_no`, `random_seq_ack_no` | Zero, relativize (first packet's seq becomes 0, subsequent are offset), or randomize TCP sequence and acknowledgment numbers. |
| **TCP options** | `zero_tcp_ts_option`, `relative_zero_tcp_ts_option`, `random_tcp_ts_option`, `zero_all_tcp_options`, `remove_all_tcp_options` | Manipulate TCP timestamp options and other TCP options. |
| **TCP window** | `zero_tcp_window`, `random_tcp_window` | Zero or randomize the TCP window size field. |
| **Checksums** | `zero_ip_checksum`, `random_ip_checksum`, `zero_transport_checksum`, `random_transport_checksum` | Zero or randomize IP/transport-layer checksums. |
| **TTL** | `zero_ip_ttl`, `random_ip_ttl` | Zero or randomize IP TTL / IPv6 hop limit. |
| **TLS SNI** | `zero_tls_sni`, `random_tls_sni` | Zero or randomize the Server Name Indication field in TLS ClientHello. |
| **Utilities** | `find_files`, `get_first_packet_timestamp`, `random_ipv4`, `random_ipv6`, `random_field`, `random_string` | File discovery, timestamp extraction, and random value generators. |

### `dataset_mm_bi_common.py` — Bidirectional Multi-Modal Features

Core feature extraction logic for **bidirectional** (session-level) flows. Imported by `feat_extract/feat_mm_bi_flows.py`.

- **`get_packet_feature(packet, header_bytes, payload_bytes, merge_packet_bytes)`** — Extracts raw byte arrays from a single packet. Separates the IP-layer header and the `Raw` payload into two integer arrays (each byte → 0–255), zero-padded to `header_bytes` and `payload_bytes` respectively.
- **`get_chunk_feature(packets, byte_packet_num, header_bytes, payload_bytes, merge_packet_bytes)`** — Processes a chunk of packets and produces three parallel sequences:
  - `bytes_list`: Raw byte arrays for the first `byte_packet_num` packets.
  - `signed_sizes`: Packet lengths with direction encoding — positive for forward (matching the first packet's source IP), negative for backward.
  - `intervals`: Inter-arrival times in seconds (6 decimal places), first packet = 0.
- **`get_flow_feature(pcap_filename, ...)`** — Reads a PCAP file, applies debias functions, optionally splits into non-overlapping chunks of `seq_packet_num` packets, and returns feature dicts per chunk.
- **`pcap_to_ft_json(pcap_dict_path, save_dir, ...)`** — Fine-tuning dataset builder. Reads a JSON file mapping `{split: {class_name: [pcap_paths]}}`, extracts features, and writes `data-train.json`, `data-valid.json`, `data-test.json`, plus `metadata.json` with the label mapping.
- **`pcap_to_pt_json(pcap_dirs, save_dir, ...)`** — Pre-training dataset builder. Scans directories of per-class PCAP folders and produces a single `data.json` (no train/test split).

### `dataset_mm_uni_common.py` — Unidirectional Multi-Modal Features

Analogous to `dataset_mm_bi_common.py` but for **unidirectional** (5-tuple flow) traffic. The key difference is that `signed_sizes` is replaced by `sizes` (always positive, since all packets share the same direction in a unidirectional flow). Exports the same `pcap_to_ft_json` and `pcap_to_pt_json` entry points.

## Data Processing Pipeline

The end-to-end pipeline consists of four sequential stages:

```
Raw PCAPs ──► pcap_split ──► pcap_clean ──► pcap_sample ──► feat_extract ──► JSON Dataset
  (Stage 1)     (Stage 2)      (Stage 3)      (Stage 4)
```

### Stage 1: PCAP Splitting (`pcap_split/`)

Splits large raw PCAP captures into individual per-flow PCAP files. Two splitting backends are used:

- **SplitCap** (via Mono): Used in `*_bi.py` scripts. Supports `session` (bidirectional, matching forward+reverse flows) and `flow` (unidirectional, strict 5-tuple) granularity.
- **Five-tuple splitter**: Used in the older `split_iscx_vpn2016.py`, `split_ustc_tfc2016.py`, and `split_cic_iot2022.py`.

Each script handles dataset-specific directory layouts and label mappings. For example, `split_iscx_vpn2016_bi.py` maps raw PCAP filenames to semantic categories (browsing, email, chat, streaming, file, voip, p2p) for both VPN and NonVPN traffic.

### Stage 2: PCAP Cleaning (`pcap_clean/`)

`clean_pcap.py` filters out flows that contain non-application-layer protocols which are irrelevant to traffic classification:

- **Removed protocols**: DNS, mDNS, LLMNR, NBNS, DHCP, NTP
- **Detection**: Both Scapy layer detection and port-based heuristics
- **Filtering logic**: If any packet in a flow matches a removed protocol, the entire flow is discarded; otherwise it is copied to the output directory. Produces `cleaning_stats.json` with per-class counts.

### Stage 3: Sampling & Splitting (`pcap_sample/`)

`sample_pcap.py` creates balanced train/valid/test splits from the cleaned per-flow PCAPs:

- **`sample_pcap_files`**: Random shuffle (seed=0), then split into 80% train / 10% valid / 10% test. Caps per-class samples at `maximum` (default: 2000) and requires `minimum` (default: 50).
- **`sample_pcap_files_by_time`**: Sorts flows by the first packet's timestamp before splitting. The test set contains the latest 10% of flows (temporal split), while train+valid are shuffled. This prevents temporal data leakage.
- **`sample_by_ratio`**: Samples a fixed percentage (e.g., 10%) of each class instead of a fixed count.

Output is a JSON file: `{train: {class: [paths]}, valid: {class: [paths]}, test: {class: [paths]}}`.

### Stage 4: Feature Extraction (`feat_extract/`)

Entry-point scripts that wire together debiasing strategies, feature extraction parameters, and dataset paths.

#### `feat_mm_bi_flows.py` — Bidirectional Feature Extraction

Configurable parameters:
- `BYTE_PACKET_NUM = 10` — Number of packets for byte-level features
- `SEQ_PACKET_NUM = 50` — Number of packets for sequence-level features
- `HEADER_BYTES = 80` — Max header bytes per packet
- `PAYLOAD_BYTES = 80` — Max payload bytes per packet

Debias strategy: `zero_ip_port` (zeroes all IP addresses and ports).

Supports parallel processing via `ProcessPoolExecutor`. Calls `pcap_to_ft_json` for fine-tuning datasets and `pcap_to_pt_json` for pre-training datasets (Browser + Kitsune).

#### `feat_mm_uni_flows.py` — Unidirectional Feature Extraction

Configurable parameters:
- `BYTE_PACKET_NUM = 5` — Number of packets for byte-level features
- `SEQ_PACKET_NUM = 50` — Number of packets for sequence-level features
- `HEADER_BYTES = 80` — Max header bytes per packet
- `PAYLOAD_BYTES = 240` — Max payload bytes per packet

Debias strategy: `none` (no field anonymization).

Processes multiple downstream datasets: CICIoT2022, CipherSpectrum, CSTNET-TLS1.3, ISCXVPN2016, USTC-TFC2016, with both random and time-sorted splits.

#### Output JSON Format

Each output JSON file contains a list of samples, where each sample has:

```json
{
  "data": ["72 0 64 ... 0 0 0", ...],
  "signed_sizes": "74 -60 52 ...",
  "intervals": "0 0.000123 0.001456 ...",
  "num_packet": 50,
  "label": 3,
  "name": "VPN-browsing",
  "pcap_file": "/path/to/flow.pcap"
}
```

| Field | Description |
|---|---|
| `data` | List of space-separated byte strings, one per packet (header + payload concatenated if `merge_packet_bytes=True`) |
| `signed_sizes` / `sizes` | Space-separated packet sizes; signed (positive=forward, negative=backward) for bidirectional, unsigned for unidirectional |
| `intervals` | Space-separated inter-arrival times in seconds |
| `num_packet` | Total number of packets in this flow/chunk |
| `label` | Integer class label |
| `name` | Human-readable class name |
| `pcap_file` | Source PCAP file path |

## Data Inspection (`data_inspect/`)

Auxiliary analysis scripts for understanding dataset properties:

- **`analyse_byte_ami.py`** — Computes Adjusted Mutual Information (AMI) between byte-range features and class labels. Produces top-K bar charts and heatmaps showing which byte positions are most informative for classification.
- **`analyse_seq_ami.py`** — Computes AMI for sequence-level features (packet sizes and inter-arrival times). Packet sizes are clipped to MTU (1500), inter-arrival times are normalized via sigmoid transform. Outputs per-position AMI scores as bar charts and heatmaps.
- **`analyse_flow_duration.py`** — Extracts and visualizes flow durations (start/end timestamps) across train/valid/test splits. Useful for verifying temporal split correctness.
- **`analyse_pcap_tls.py`** — Uses `tshark` to analyze TLS/SSL/QUIC protocol statistics: stream counts, packet ratios, cipher suite distributions, and TLS version breakdowns. `run_analyse_pcap_tls.sh` provides batch invocations for all datasets.

## Supported Datasets

| Dataset | Splitting Script(s) | Task |
|---|---|---|
| CIC IoT 2022 | `split_cic_iot2022*.py` | IoT device identification |
| CipherSpectrum | `split_cstnet_tls_uni.py` | Encrypted traffic classification |
| CSTNET-TLS 1.3 | `split_cstnet_tls_uni.py` | TLS 1.3 application identification |
| ISCX VPN 2016 | `split_iscx_vpn2016*.py` | VPN traffic classification |
| USTC-TFC 2016 | `split_ustc_tfc2016*.py` | Malware traffic detection |
| CrossNet 2021 | `split_crossnet_bi.py` | Cross-network traffic classification |
| Kitsune | `split_kitsune_bi.py` | Network intrusion detection |
| Browser | `split_browser_bi.py` | Web traffic classification |
| CIC Android Malware 2017 | `split_cic_andmal_bi.py` | Android malware detection |
