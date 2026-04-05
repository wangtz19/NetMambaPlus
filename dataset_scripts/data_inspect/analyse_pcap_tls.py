import subprocess
from collections import defaultdict, Counter
import pandas as pd
import os
from tqdm import tqdm
import json
from typing import List, Dict, Any
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_common import find_files


def run_tshark(command_args):
    """Run a tshark subprocess and return output lines."""
    result = subprocess.run(command_args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    return result.stdout.strip().splitlines()


def get_all_tcp_streams(pcap_file): # only tcp streams
    """Return set of all TCP stream indexes in the pcap."""
    cmd = ["tshark", "-r", pcap_file, "-T", "fields", "-e", "tcp.stream"]
    output = run_tshark(cmd)
    return set(line for line in output if line.strip() != "")


def get_all_udp_streams(pcap_file): # only udp streams
    """Return set of all UDP stream indexes in the pcap."""
    cmd = ["tshark", "-r", pcap_file, "-T", "fields", "-e", "udp.stream"]
    output = run_tshark(cmd)
    return set(line for line in output if line.strip() != "")


def get_tls_ssl_streams(pcap_file):
    """Return set of all TLS/SSL stream indexes in the pcap."""
    cmd = ["tshark", "-r", pcap_file, "-Y", "ssl || tls", "-T", "fields", "-e", "tcp.stream"]
    output = run_tshark(cmd)
    return set(line for line in output if line.strip() != "")


def get_quic_streams(pcap_file):
    """Return set of all QUIC stream indexes in the pcap."""
    cmd = ["tshark", "-r", pcap_file, "-Y", "quic || gquic", "-T", "fields", "-e", "udp.stream"]
    output = run_tshark(cmd)
    return set(line for line in output if line.strip() != "")


def get_tls_streams_ciphers_versions(pcap_file):
    """Return dict of stream_id -> (cipher_suite, tls_version, supported_version) tuples."""
    cmd = [
        "tshark", "-r", pcap_file,
        "-Y", "tls.handshake.ciphersuite",
        "-T", "fields",
        "-e", "tcp.stream",
        "-e", "tls.handshake.type", # 1 for ClientHello, 2 for ServerHello
        "-e", "tls.handshake.ciphersuite",
        "-e", "tls.handshake.version",
        "-e", "tls.handshake.extensions.supported_version"
    ]
    output = run_tshark(cmd)

    stream_info = defaultdict(list)
    for line in output:
        parts = line.strip().split('\t')
        if len(parts) == 5:
            stream_id, handshake_type, cipher, version, supported_version = parts
            try:
                if int(handshake_type) == 2: # ciphersuite is determined in ServerHello
                    stream_info[stream_id].append((cipher, version, supported_version))
            except Exception as e:
                print(f"Error processing line: {line.strip()} - {e}")
        if len(parts) == 4:
            stream_id, handshake_type, cipher, version = parts
            try:
                if int(handshake_type) == 2:
                    stream_info[stream_id].append((cipher, version, None))
            except Exception as e:
                print(f"Error processing line: {line.strip()} - {e}")
    return stream_info


def get_ssl_streams_versions(pcap_file):
    """Return dict of stream_id -> (ssl_version) tuples."""
    cmd = [
        "tshark", "-r", pcap_file,
        "-Y", "ssl.handshake.version",
        "-T", "fields",
        "-e", "tcp.stream",
        "-e", "ssl.handshake.version"
    ]
    output = run_tshark(cmd)

    stream_info = defaultdict(list)
    for line in output:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            stream_id, version = parts
            stream_info[stream_id].append((version,))
    return stream_info


def count_total_packets(pcap_file): # including tcp and udp packets
    cmd = ["tshark", "-r", pcap_file, "-T", "fields", "-e", "frame.number"]
    output = run_tshark(cmd)
    return len(output)


def count_tls_ssl_packets(pcap_file): # count only TLS packets
    cmd = ["tshark", "-r", pcap_file, "-Y", "tls || ssl", "-T", "fields", "-e", "frame.number"]
    output = run_tshark(cmd)
    return len(output)


def count_quic_packets(pcap_file): # count only QUIC packets
    cmd = ["tshark", "-r", pcap_file, "-Y", "quic || gquic", "-T", "fields", "-e", "frame.number"]
    output = run_tshark(cmd)
    return len(output)


def count_packet_protocols(pcap_file):
    """Count packets by protocol type."""
    cmd = ["tshark", "-r", pcap_file, "-T", "fields", "-e", "_ws.col.Protocol"]
    output = run_tshark(cmd)
    protocol_counter = Counter()
    for line in output:
        protocols = line.strip().split(',')
        for protocol in protocols:
            if protocol:
                protocol_counter[protocol] += 1
    return protocol_counter


def map_tls_version(version, supported_version):
    if supported_version == "0x0304":
        return "TLS 1.3"
    elif version == "0x0303":
        return "TLS 1.2"
    elif version == "0x0302":
        return "TLS 1.1"
    elif version == "0x0301":
        return "TLS 1.0"
    else:
        return f"Unknown ({supported_version})" if supported_version else "Unknown (no version)"


def get_cipher_map():
    id_to_cipher = {}
    cipher_df = pd.read_csv("id-cipher.csv", header=None, names=["id", "cipher"])
    for _, row in cipher_df.iterrows():
        id_to_cipher[int(row['id'])] = row['cipher']
    return id_to_cipher


def map_tls_cipher(cipher_hex, cipher_map):
    cipher_id = int(cipher_hex, 16) if cipher_hex.startswith("0x") else int(cipher_hex)
    return cipher_map.get(cipher_id, f"Unknown ({cipher_hex})")


def process_pcap(pcap_file, verbose=False):
    if verbose:
        print(f"🔍 Processing pcap file: {pcap_file}")

    all_tcp_streams = get_all_tcp_streams(pcap_file)
    all_udp_streams = get_all_udp_streams(pcap_file)
    tls_ssl_streams = get_tls_ssl_streams(pcap_file)
    quic_streams = get_quic_streams(pcap_file)

    tls_stream_info = get_tls_streams_ciphers_versions(pcap_file) # per stream
    ssl_stream_info = get_ssl_streams_versions(pcap_file) # per stream

    total_packets = count_total_packets(pcap_file)
    total_tls_packets = count_tls_ssl_packets(pcap_file)
    total_quic_packets = count_quic_packets(pcap_file)

    protocol_counter = count_packet_protocols(pcap_file)

    if verbose:
        print("\n📊 Encrypted Stream Statistics")
        total_streams = len(all_tcp_streams) + len(all_udp_streams)
        total_encrypted_streams = len(tls_ssl_streams) + len(quic_streams)
        print(f"Total streams: {total_streams}")
        print(f"TLS/SSL streams: {len(tls_ssl_streams)}")
        print(f"QUIC streams: {len(quic_streams)}")
        print(f"Encrypted stream ratio: {total_encrypted_streams/total_streams:.3%}" if total_streams > 0 else "N/A")

        print("\n📦 Encrypted Packet Statistics")
        print(f"Total packets: {total_packets}")
        print(f"TLS/SSL packets: {total_tls_packets}")
        print(f"QUIC packets: {total_quic_packets}")
        print(f"Encrypted packet ratio: {(total_tls_packets+total_quic_packets) / total_packets:.3%}" if total_packets > 0 else "N/A")

    tls_cipher_counter = Counter()
    tls_ssl_version_counter = Counter()

    cipher_map = get_cipher_map()
    for stream_id, items in tls_stream_info.items():
        for cipher, version, supported_version in items:
            tls_cipher_counter[map_tls_cipher(cipher, cipher_map)] += 1
            tls_ssl_version_counter[map_tls_version(version, supported_version)] += 1
    for stream_id, items in ssl_stream_info.items():
        for version in items:
            tls_ssl_version_counter[map_tls_version(version[0], None)] += 1

    if verbose:
        print("\n🔐 Frequency of TLS ciphers per stream")
        for cipher, count in tls_cipher_counter.items():
            print(f"{cipher}: {count} times")

        print("\n📈 Frequency of TLS/SSL version per stream")
        for version, count in tls_ssl_version_counter.items():
            print(f"{version}: {count} times")

        print("\n🔍 Protocols in the pcap file")
        for protocol, count in protocol_counter.items():
            print(f"{protocol}: {count} packets")

        print("\n✅ Finish processing pcap file")

    return {
        "total_streams": len(all_tcp_streams) + len(all_udp_streams),
        "tls_ssl_streams": len(tls_ssl_streams),
        "quic_streams": len(quic_streams),
        "total_packets": total_packets,
        "tls_ssl_packets": total_tls_packets,
        "quic_packets": total_quic_packets,
        "tls_cipher_counter": tls_cipher_counter,
        "tls_ssl_version_counter": tls_ssl_version_counter,
        "protocol_counter": protocol_counter
    }


def process_dirs(input_dirs: List[str], output_file: str, verbose: bool = False):
    pcap_files = []
    for input_dir in input_dirs:
        pcap_files.extend(find_files(input_dir, extension=".pcap"))
    total_streams = 0
    total_tls_ssl_streams = 0
    total_quic_streams = 0
    total_packets = 0
    total_tls_ssl_packets = 0
    total_quic_packets = 0
    tls_cipher_counter = Counter()
    tls_ssl_version_counter = Counter()
    protocol_counter = Counter()
    for pcap_file in tqdm(pcap_files, desc="Processing pcap files"):
        result = process_pcap(pcap_file, verbose=verbose)
        total_streams += result["total_streams"]
        total_tls_ssl_streams += result["tls_ssl_streams"]
        total_quic_streams += result["quic_streams"]
        total_packets += result["total_packets"]
        total_tls_ssl_packets += result["tls_ssl_packets"]
        total_quic_packets += result["quic_packets"]
        tls_cipher_counter.update(result["tls_cipher_counter"])
        tls_ssl_version_counter.update(result["tls_ssl_version_counter"])
        protocol_counter.update(result["protocol_counter"])
    summary = {
        "total_streams": total_streams,
        "tls_ssl_streams": total_tls_ssl_streams,
        "quic_streams": total_quic_streams,
        "encypted_stream_ratio": round((total_tls_ssl_streams + total_quic_streams) / total_streams, 4) if total_streams > 0 else "N/A",
        "total_packets": total_packets,
        "tls_ssl_packets": total_tls_ssl_packets,
        "quic_packets": total_quic_packets,
        "encypted_packet_ratio": round((total_tls_ssl_packets + total_quic_packets) / total_packets, 4) if total_packets > 0 else "N/A",
        "tls_cipher_counter": dict(tls_cipher_counter),
        "tls_ssl_version_counter": dict(tls_ssl_version_counter),
        "protocol_counter": dict(protocol_counter)
    }
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze pcap files for TLS/SSL and QUIC streams.")
    parser.add_argument("--input_dirs", nargs='+', help="Directories containing pcap files to analyze.")
    parser.add_argument("--output_file", type=str, default="pcap_analysis_summary.json",
                        help="Output file to save the analysis summary.")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose output.")
    
    args = parser.parse_args()
    print(args)
    process_dirs(args.input_dirs, args.output_file, verbose=args.verbose)
    print(f"Analysis summary saved to {args.output_file}")
