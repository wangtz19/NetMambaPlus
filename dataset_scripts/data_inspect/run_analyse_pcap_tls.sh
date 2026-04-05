set -x

# python analyse_pcap_tls.py \
#     --input_dirs /mnt/ssd1/wtz_nta_dataset/CrossNet2021/ScenarioA \
#     --output_file /root/Vim/dataset_scripts/data_analysis/pcap_tls/CrossNet2021-ScenarioA.json \
#     > /root/Vim/dataset_scripts/data_analysis/pcap_tls/CrossNet2021-ScenarioA.log 2>&1 &

# python analyse_pcap_tls.py \
#     --input_dirs /mnt/ssd1/wtz_nta_dataset/CrossNet2021/ScenarioB \
#     --output_file /root/Vim/dataset_scripts/data_analysis/pcap_tls/CrossNet2021-ScenarioB.json \
#     > /root/Vim/dataset_scripts/data_analysis/pcap_tls/CrossNet2021-ScenarioB.log 2>&1 &

# python analyse_pcap_tls.py \
#     --input_dirs /mnt/ssd1/wtz_nta_dataset/CipherSpectrum/bi-flows/mix \
#     --output_file /root/Vim/dataset_scripts/data_analysis/pcap_tls/CipherSpectrum.json \
#     > /root/Vim/dataset_scripts/data_analysis/pcap_tls/CipherSpectrum.log 2>&1 &

# python analyse_pcap_tls.py \
#     --input_dirs /mnt/ssd1/wtz_nta_dataset/CSTNET-TLS1.3/bi-flows \
#     --output_file /root/Vim/dataset_scripts/data_analysis/pcap_tls/CSTNET-TLS1.3.json \
#     > /root/Vim/dataset_scripts/data_analysis/pcap_tls/CSTNET-TLS1.3.log 2>&1 &

# python analyse_pcap_tls.py \
#     --input_dirs /mnt/ssd1/wtz_nta_dataset/ISCXVPN2016/VPN \
#     --output_file /root/Vim/dataset_scripts/data_analysis/pcap_tls/ISCXVPN2016.json \
#     > /root/Vim/dataset_scripts/data_analysis/pcap_tls/ISCXVPN2016.log 2>&1 &

# python analyse_pcap_tls.py \
#     --input_dirs /mnt/ssd1/wtz_nta_dataset/USTC-TFC2016/Benign /mnt/ssd1/wtz_nta_dataset/USTC-TFC2016/Malware \
#     --output_file /root/Vim/dataset_scripts/data_analysis/pcap_tls/USTC-TFC2016.json \
#     > /root/Vim/dataset_scripts/data_analysis/pcap_tls/USTC-TFC2016.log 2>&1 &

# python analyse_pcap_tls.py \
#     --input_dirs /mnt/ssd1/wtz_nta_dataset/CICIoT2022/1-Power /mnt/ssd1/wtz_nta_dataset/CICIoT2022/6-Attacks \
#     --output_file /root/Vim/dataset_scripts/data_analysis/pcap_tls/CICIoT2022.json \
#     > /root/Vim/dataset_scripts/data_analysis/pcap_tls/CICIoT2022.log 2>&1 &

# python analyse_pcap_tls.py \
#     --input_dirs /mnt/ssd1/wtz_nta_dataset/Browser/pcap \
#     --output_file /root/Vim/dataset_scripts/data_analysis/pcap_tls/Browser.json \
#     > /root/Vim/dataset_scripts/data_analysis/pcap_tls/Browser.log 2>&1 &

# python analyse_pcap_tls.py \
#     --input_dirs /mnt/ssd1/wtz_nids_dataset/kitsune/pcap \
#     --output_file /root/Vim/dataset_scripts/data_analysis/pcap_tls/kitsune.json \
#     > /root/Vim/dataset_scripts/data_analysis/pcap_tls/kitsune.log 2>&1 &

# python analyse_pcap_tls.py \
#     --input_dirs /mnt/ssd1/wtz_nta_dataset/CrossPlatform/pcap/china/android /mnt/ssd1/wtz_nta_dataset/CrossPlatform/pcap/india/android /mnt/ssd1/wtz_nta_dataset/CrossPlatform/pcap/us/android \
#     --output_file /root/Vim/dataset_scripts/data_analysis/pcap_tls/CrossPlatform-Andorid.json \
#     > /root/Vim/dataset_scripts/data_analysis/pcap_tls/CrossPlatform-Andorid.log 2>&1 &

# python analyse_pcap_tls.py \
#     --input_dirs /mnt/ssd1/wtz_nta_dataset/CrossPlatform/pcap/china/ios /mnt/ssd1/wtz_nta_dataset/CrossPlatform/pcap/india/ios /mnt/ssd1/wtz_nta_dataset/CrossPlatform/pcap/us/ios \
#     --output_file /root/Vim/dataset_scripts/data_analysis/pcap_tls/CrossPlatform-iOS.json \
#     > /root/Vim/dataset_scripts/data_analysis/pcap_tls/CrossPlatform-iOS.log 2>&1 &

# python analyse_pcap_tls.py \
#     --input_dirs /mnt/ssd1/wtz_nta_dataset/DataCon2021-Proxy/datacon2021_eta/part1/real_data /mnt/ssd1/wtz_nta_dataset/DataCon2021-Proxy/datacon2021_eta/part1/sample \
#     --output_file /root/Vim/dataset_scripts/data_analysis/pcap_tls/datacon2021_p1.json \
#     > /root/Vim/dataset_scripts/data_analysis/pcap_tls/datacon2021_p1.log 2>&1 &


python analyse_pcap_tls.py \
    --input_dirs /mnt/ssd1/wtz_nta_dataset/ISCXVPN2016/NonVPN \
    --output_file /root/Vim/dataset_scripts/data_analysis/pcap_tls/ISCX-NonVPN2016.json \
    > /root/Vim/dataset_scripts/data_analysis/pcap_tls/ISCX-NonVPN2016.log 2>&1 &