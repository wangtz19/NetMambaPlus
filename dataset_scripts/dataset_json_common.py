import json
import random
import os
import binascii
import numpy as np
import scapy.all as scapy
from tqdm import tqdm
from typing import Callable, Union


def find_files(data_path: str, extension: str=".pcap"):
    pcap_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(extension):
                pcap_files.append(os.path.join(root, file))
    return pcap_files



def get_first_packet_timestamp(pcap_file):
    try:
        packets = scapy.rdpcap(pcap_file, count=1)
        if packets:
            return packets[0].time
        return float("inf")
    except Exception as e:
        return float("inf")


def zero_ip(packets: scapy.PacketList):
    # set src_ip and dst_ip to
    for packet in packets:
        if packet.haslayer("IP"):
            packet["IP"].src = "0.0.0.0"
        if packet.haslayer("IPv6"):
            packet["IPv6"].src = "0:0:0:0:0:0:0:0"
    return packets


def zero_port(packets: scapy.PacketList):
    # set src_port and dst_port to 0
    for packet in packets:
        if packet.haslayer("TCP"):
            packet["TCP"].sport = 0
            packet["TCP"].dport = 0
        if packet.haslayer("UDP"):
            packet["UDP"].sport = 0
            packet["UDP"].dport = 0
    return packets


def zero_ip_port(packets: scapy.PacketList):
    # set src_ip, dst_ip, src_port, dst_port to 0
    for packet in packets:
        if packet.haslayer("IP"):
            packet["IP"].src = "0.0.0.0"
            packet["IP"].dst = "0.0.0.0"
        if packet.haslayer("IPv6"):
            packet["IPv6"].src = "0:0:0:0:0:0:0:0"
            packet["IPv6"].dst = "0:0:0:0:0:0:0:0"
        if packet.haslayer("TCP"):
            packet["TCP"].sport = 0
            packet["TCP"].dport = 0
        if packet.haslayer("UDP"):
            packet["UDP"].sport = 0
            packet["UDP"].dport = 0
    return packets


def relative_zero_ip(packets: scapy.PacketList):
    first_src_ip = None
    for packet in packets:
        if packet.haslayer("IP"):
            if first_src_ip is None:
                first_src_ip = packet["IP"].src
            if packet["IP"].src == first_src_ip:
                packet["IP"].src = "0.0.0.0"
                packet["IP"].dst = "0.0.0.1"
            else:
                packet["IP"].src = "0.0.0.1"
                packet["IP"].dst = "0.0.0.0"
        if packet.haslayer("IPv6"):
            if first_src_ip is None:
                first_src_ip = packet["IPv6"].src
            if packet["IPv6"].src == first_src_ip:
                packet["IPv6"].src = "0:0:0:0:0:0:0:0"
                packet["IPv6"].dst = "0:0:0:0:0:0:0:1"
            else:
                packet["IPv6"].src = "0:0:0:0:0:0:0:1"
                packet["IPv6"].dst = "0:0:0:0:0:0:0:0"
    return packets


def relative_zero_port(packets: scapy.PacketList):
    first_src_port = None
    for packet in packets:
        if packet.haslayer("TCP"):
            if first_src_port is None:
                first_src_port = packet["TCP"].sport
            if packet["TCP"].sport == first_src_port:
                packet["TCP"].sport = 0
                packet["TCP"].dport = 1
            else:
                packet["TCP"].sport = 1
                packet["TCP"].dport = 0
        if packet.haslayer("UDP"):
            if first_src_port is None:
                first_src_port = packet["UDP"].sport
            if packet["UDP"].sport == first_src_port:
                packet["UDP"].sport = 0
                packet["UDP"].dport = 1
            else:
                packet["UDP"].sport = 1
                packet["UDP"].dport = 0
    return packets


# generate random ipv4 address
def random_ipv4():
    IPV4_MAX = ipaddress.IPv4Address._ALL_ONES # type: ignore
    ip_int = random.randint(0, IPV4_MAX)
    ip_str = ipaddress.IPv4Address._string_from_ip_int(ip_int) # type: ignore
    return ip_str


# generate random ipv6 address
def random_ipv6():
    IPV6_MAX = ipaddress.IPv6Address._ALL_ONES # type: ignore
    ip_int = random.randint(0, IPV6_MAX)
    ip_str = ipaddress.IPv6Address._string_from_ip_int(ip_int) # type: ignore
    return ip_str


def random_field(bits):
    field_max = 2**bits-1
    field_int = random.randint(0, field_max)
    return field_int


def random_string(length):
    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._"
    return ''.join(random.choice(characters) for _ in range(length))


def random_ip_port(packets: scapy.PacketList):
    first_packet = packets[0]
    if first_packet.haslayer("IP"):
        first_src_ip = first_packet["IP"].src
        rsrc_ip = random_ipv4()
        rdst_ip = random_ipv4()
        ip_key = "IP"
    elif first_packet.haslayer("IPv6"):
        first_src_ip = first_packet["IPv6"].src
        rsrc_ip = random_ipv6()
        rdst_ip = random_ipv6()
        ip_key = "IPv6"
    else:
        raise ValueError("No IP layer found in the first packet")
    rsrc_port = random_field(16)
    rdst_port = random_field(16)
    if first_packet.haslayer("TCP"):
        first_src_port = first_packet["TCP"].sport
        transport_key = "TCP"
    elif first_packet.haslayer("UDP"):
        first_src_port = first_packet["UDP"].sport
        transport_key = "UDP"
    else:
        raise ValueError("No transport layer found in the first packet")
    for packet in packets:
        if packet.haslayer(ip_key):
            if packet[ip_key].src == first_src_ip:
                packet[ip_key].src = rsrc_ip
                packet[ip_key].dst = rdst_ip
            else:
                packet[ip_key].src = rdst_ip
                packet[ip_key].dst = rsrc_ip
        if packet.haslayer(transport_key):
            if packet[transport_key].sport == first_src_port:
                packet[transport_key].sport = rsrc_port
                packet[transport_key].dport = rdst_port
            else:
                packet[transport_key].sport = rdst_port
                packet[transport_key].dport = rsrc_port
    return packets


def zero_tls_sni(packets: scapy.PacketList):
    from scapy.layers.tls.all import TLS, TLSClientHello, TLS_Ext_ServerName
    scapy.load_layer("tls")
    for packet in packets:
        if packet.haslayer(TLS):
            tls = packet[TLS]
            if TLSClientHello in tls:
                client_hello = tls[TLSClientHello]
                if client_hello is not None and hasattr(client_hello, "ext") \
                    and client_hello.ext is not None:
                    for ext in client_hello.ext:
                        if isinstance(ext, TLS_Ext_ServerName):
                            for sn in ext.servernames: # type: ignore
                                if hasattr(sn, "servername") and sn.namelen > 0:
                                    sn.servername = ("0" * sn.namelen).encode("UTF-8")
                                    packet[TLS] = TLS(bytes(tls)) # force the TLS layer to be re-encoded to update SNI
    return packets


def random_tls_sni(packets: scapy.PacketList):
    from scapy.layers.tls.all import TLS, TLSClientHello, TLS_Ext_ServerName
    scapy.load_layer("tls")
    for packet in packets:
        if packet.haslayer(TLS):
            tls = packet[TLS]
            if TLSClientHello in tls:
                client_hello = tls[TLSClientHello]
                if client_hello is not None and hasattr(client_hello, "ext") \
                    and client_hello.ext is not None:
                    for ext in client_hello.ext:
                        if isinstance(ext, TLS_Ext_ServerName):
                            for sn in ext.servernames:  # type: ignore
                                if hasattr(sn, "servername") and sn.namelen > 0:
                                    sn.servername = random_string(sn.namelen).encode("UTF-8")
                                    packet[TLS] = TLS(bytes(tls)) # force the TLS layer to be re-encoded to update SNI
    return packets


def zero_tcp_window(packets: scapy.PacketList):
    # set TCP window size to 0
    for packet in packets:
        if packet.haslayer("TCP"):
            packet["TCP"].window = 0
    return packets


def random_tcp_window(packets: scapy.PacketList):
    # set TCP window size to a random value
    for packet in packets:
        if packet.haslayer("TCP"):
            packet["TCP"].window = random_field(16)  # 16 bits for TCP window size
    return packets


def zero_tcp_ts_option(packets: scapy.PacketList):
    # set TCP timestamp option to 0
    for packet in packets:
        if packet.haslayer("TCP"):
            ts_option = packet["TCP"].options
            for i, opt in enumerate(ts_option):
                if opt[0] == "Timestamp":
                    ts_option[i] = ("Timestamp", (0, 0))
    return packets


def zero_all_tcp_options(packets: scapy.PacketList, remove_options: bool = False):
    # set all TCP options to 0
    for packet in packets:
        if packet.haslayer("TCP"):
            # get the total bytes of TCP options, fill with all 0
            opt_len = packet["TCP"].dataofs * 4 - 20
            if opt_len > 0:
                if remove_options:
                    packet["TCP"].options = []
                    packet["TCP"].dataofs = 5  # reset data offset to minimum
                    packet["IP"].len = packet["IP"].len - opt_len  # adjust IP total length
                else:
                    packet["TCP"].options = [("EOL", None)] * (opt_len)
    return packets


remove_all_tcp_options = partial(zero_all_tcp_options, remove_options=True)


def relative_zero_tcp_ts_option(packets: scapy.PacketList):
    """
    Safely convert TCP Timestamp option to relative values.
    Only modify options that have the expected format (("Timestamp", (tsval, tsecr)))
    and where tsval/tsecr are parseable as integers. Leave malformed options unchanged.
    """
    MAX_UINT32 = 2**32 - 1
    first_src_ts = None
    first_dst_ts = None
    src_port = None

    for packet in packets:
        if not packet.haslayer("TCP"):
            continue
        try:
            orig_opts = packet["TCP"].options or []
        except Exception:
            continue
        tcp_options = []
        modified = False
        for opt in orig_opts:
            try:
                # normalize option to (name, value) pair if possible
                if (isinstance(opt, (tuple, list))) and len(opt) >= 1:
                    name = opt[0]
                    val = opt[1] if len(opt) > 1 else None
                else:
                    # unknown format, keep as-is
                    tcp_options.append(opt)
                    continue
                if name != "Timestamp":
                    tcp_options.append(opt)
                    continue
                # parse timestamp pair robustly
                ts_val = None
                ts_ecr = None
                if isinstance(val, (tuple, list)) and len(val) >= 2:
                    ts_val = val[0]
                    ts_ecr = val[1]
                else:
                    # try to parse bytes -> two 4-byte integers (big endian)
                    if isinstance(val, (bytes, bytearray)) and len(val) >= 8:
                        try:
                            ts_val = int.from_bytes(val[0:4], byteorder="big", signed=False)
                            ts_ecr = int.from_bytes(val[4:8], byteorder="big", signed=False)
                        except Exception:
                            ts_val = None
                            ts_ecr = None
                # if cannot parse integers, keep original option to avoid malformed encoding
                if ts_val is None and ts_ecr is None:
                    tcp_options.append(opt)
                    continue

                # set defaults when one side is missing
                if ts_val is None:
                    ts_val = 0
                if ts_ecr is None:
                    ts_ecr = 0

                # initialize src_port and first timestamps
                if src_port is None:
                    src_port = packet["TCP"].sport
                    first_src_ts = int(ts_val) if ts_val is not None else 0
                    first_dst_ts = int(ts_ecr) if ts_ecr is not None else 0

                # compute relative values depending on direction
                if packet["TCP"].sport == src_port:  # forward
                    new_ts_val = int(ts_val) - int(first_src_ts) if ts_val is not None else 0
                    new_ts_ecr = int(ts_ecr) - int(first_dst_ts) if (ts_ecr is not None and first_dst_ts is not None) else 0
                else:  # backward
                    if first_dst_ts is None:
                        first_dst_ts = int(ts_val) if ts_val is not None else 0
                    new_ts_val = int(ts_val) - int(first_dst_ts) if ts_val is not None else 0
                    new_ts_ecr = int(ts_ecr) - int(first_src_ts) if (ts_ecr is not None and first_src_ts is not None) else 0

                # clamp into valid uint32 range and ensure int
                new_ts_val = max(0, min(MAX_UINT32, int(new_ts_val)))
                new_ts_ecr = max(0, min(MAX_UINT32, int(new_ts_ecr)))

                # append the safe, normalized option
                tcp_options.append(("Timestamp", (new_ts_val, new_ts_ecr)))
                modified = True
            except Exception:
                # on any parsing/processing error, keep original opt
                tcp_options.append(opt)

        # attempt to write back modified options; if fails, keep original to avoid raising
        if modified:
            try:
                packet["TCP"].options = tcp_options
            except Exception:
                # fallback: keep original options
                packet["TCP"].options = orig_opts

    return packets


def random_tcp_ts_option(packets: scapy.PacketList):
    # set TCP timestamp option to relative time where the first packet is 0
    first_src_ts, first_dst_ts = None, None
    src_port = None
    rsrc_ts_offset = random_field(16) # though TCP timestamp is 32 bits, we use 16 bits to avoid overflow
    rdst_ts_offset = rsrc_ts_offset + random_field(5)
    for packet in packets:
        if packet.haslayer("TCP"):
            tcp_options = [list(option) for option in packet["TCP"].options]
            for option in tcp_options:
                if option[0] == "Timestamp":
                    if src_port == None:
                        src_port = packet["TCP"].sport
                        first_src_ts = option[1][0]
                        if option[1][1] != 0:
                            first_dst_ts = option[1][1]
                    if packet["TCP"].sport == src_port: # forward
                        if option[1][1] != 0:
                            option[1] = (option[1][0] - first_src_ts + rsrc_ts_offset, 
                                         option[1][1] - first_dst_ts + rdst_ts_offset)
                        else:
                            option[1] = (option[1][0] - first_src_ts + rsrc_ts_offset,
                                        rdst_ts_offset)
                    else: # backward
                        if first_dst_ts == None:
                            first_dst_ts = option[1][0]
                        if option[1][1] != 0:
                            option[1] = (option[1][0] - first_dst_ts + rdst_ts_offset, 
                                         option[1][1] - first_src_ts + rsrc_ts_offset)
                        else:
                            option[1] = (option[1][0] - first_dst_ts + rdst_ts_offset, 
                                         rsrc_ts_offset)
            packet["TCP"].options = [tuple(option) for option in tcp_options]
    return packets


def zero_ip_checksum(packets: scapy.PacketList):
    # set IP checksum to 0
    for packet in packets:
        if packet.haslayer("IP"):
            packet["IP"].chksum = 0
        if packet.haslayer("IPv6"):
            packet["IPv6"].chksum = 0
    return packets


def random_ip_checksum(packets: scapy.PacketList):
    # set IP checksum to a random value
    for packet in packets:
        if packet.haslayer("IP"):
            packet["IP"].chksum = random_field(16)  # 16 bits for IP checksum
        if packet.haslayer("IPv6"):
            packet["IPv6"].chksum = random_field(16)  # 16 bits for IPv6 checksum
    return packets


def zero_transport_checksum(packets: scapy.PacketList):
    # set TCP checksum to 0
    for packet in packets:
        if packet.haslayer("TCP"):
            packet["TCP"].chksum = 0
        if packet.haslayer("UDP"):
            packet["UDP"].chksum = 0
    return packets


def random_transport_checksum(packets: scapy.PacketList):
    # set TCP checksum to a random value
    for packet in packets:
        if packet.haslayer("TCP"):
            packet["TCP"].chksum = random_field(16)  # 16 bits for TCP checksum
        if packet.haslayer("UDP"):
            packet["UDP"].chksum = random_field(16)  # 16 bits for UDP checksum
    return packets


def zero_ip_ttl(packets: scapy.PacketList):
    # set IP TTL to 0
    for packet in packets:
        if packet.haslayer("IP"):
            packet["IP"].ttl = 0
        if packet.haslayer("IPv6"):
            packet["IPv6"].hlim = 0
    return packets


def random_ip_ttl(packets: scapy.PacketList):
    # set IP TTL to a random value
    for packet in packets:
        if packet.haslayer("IP"):
            packet["IP"].ttl = random_field(8)  # 8 bits for IP TTL
        if packet.haslayer("IPv6"):
            packet["IPv6"].hlim = random_field(8)  # 8 bits for IPv6 hop limit
    return packets


def zero_seq_ack_no(packets: scapy.PacketList):
    # set TCP seq and ack no to 0
    for packet in packets:
        if packet.haslayer("TCP"):
            packet["TCP"].seq = 0
            packet["TCP"].ack = 0
    return packets


def relative_zero_seq_ack_no(packets: scapy.PacketList):
    # set TCP seq and ack no to relative time where the first packet is 0
    first_src_seq, first_src_ack = None, None
    first_dst_seq = None
    src_port = None
    for packet in packets: # first pass to get the first forward and backward seq
        if packet.haslayer("TCP"):
            if src_port == None:
                src_port = packet["TCP"].sport
                first_src_seq = packet["TCP"].seq
                first_src_ack = packet["TCP"].ack
            else:
                if first_dst_seq == None:
                    first_dst_seq = packet["TCP"].seq
    if first_dst_seq == None:
        first_dst_seq = first_src_ack
    for idx, packet in enumerate(packets): # second pass to set the seq and ack no
        if packet.haslayer("TCP"):
            if src_port == packet["TCP"].sport: # forward
                packet["TCP"].seq = (packet["TCP"].seq - first_src_seq) % (2**32)
                if idx != 0: # set ack no
                    packet["TCP"].ack = (packet["TCP"].ack - first_dst_seq) % (2**32)
                else: # first packet with ack set to 0
                    packet["TCP"].ack = 0
            else: # backward
                packet["TCP"].seq = (packet["TCP"].seq - first_dst_seq) % (2**32)
                packet["TCP"].ack = (packet["TCP"].ack - first_src_seq) % (2**32)
    return packets


def random_seq_ack_no(packets: scapy.PacketList):
    # set TCP seq and ack no to random values
    first_src_seq, first_src_ack = None, None
    first_dst_seq = None
    src_port = None
    for packet in packets: # first pass to get the first forward and backward seq
        if packet.haslayer("TCP"):
            if src_port == None:
                src_port = packet["TCP"].sport
                first_src_seq = packet["TCP"].seq
                first_src_ack = packet["TCP"].ack
            else:
                if first_dst_seq == None:
                    first_dst_seq = packet["TCP"].seq
    if first_dst_seq == None:
        first_dst_seq = first_src_ack
    rsrc_seq_offset = random_field(32) # 32 bits for TCP seq no
    rdst_seq_offset = rsrc_seq_offset + random_field(5)
    for idx, packet in enumerate(packets): # second pass to set the seq and ack no
        if packet.haslayer("TCP"):
            if src_port == packet["TCP"].sport: # forward
                packet["TCP"].seq = (packet["TCP"].seq - first_src_seq + rsrc_seq_offset) % (2**32)
                if idx != 0: # set ack no
                    packet["TCP"].ack = (packet["TCP"].ack - first_dst_seq + rdst_seq_offset) % (2**32)
                else: # first packet with ack set to 0
                    packet["TCP"].ack = rdst_seq_offset
            else: # backward
                packet["TCP"].seq = (packet["TCP"].seq - first_dst_seq + rdst_seq_offset) % (2**32)
                packet["TCP"].ack = (packet["TCP"].ack - first_src_seq + rsrc_seq_offset) % (2**32)
    return packets
