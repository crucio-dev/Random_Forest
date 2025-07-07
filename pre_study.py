import pandas as pd
import os
import glob
import gc

# from count_benign import count_benign_packets
# from sample_collecting import sample_collecting
from final_sample import final_sample

# Replace with your folder path
folder_path = '../../DDOS DataSet/CSV/Total'  # Specify your folder path here

# Constants
RANDOM_SEED = 42

# all_files, benign_count_total, benign_packets_df = count_benign_packets()
# print(f"Total benign packets: {len(benign_packets_df)}")

benign_count_total = 7300017
all_files = glob.glob(os.path.join(folder_path, '*.csv'))
attack_types = ['DrDoS_SSDP', 'NetBIOS', 'DDoS-LOIC-HTTP', 'LDAP', 'DrDoS_SNMP', 'DrDoS_LDAP', 'Portmap', 'DrDoS_MSSQL', 'DrDoS_DNS', 'DrDoS_NetBIOS', 'DrDoS_UDP', 'DrDoS_NTP', 'UDP', 'MSSQL', 'TFTP', 'UDP-lag', 'Syn', 'DoS-Hulk']
attack_type_count = 18

samples_per_attack_type = benign_count_total // attack_type_count
print(f"Samples per attack type: {samples_per_attack_type}")

final_sample(attack_types, samples_per_attack_type, RANDOM_SEED)  # Collect and sample attack packets

"""
print("Loading benign packets with dtype dictionary...")
dtype_df = pd.read_csv('benign_packets_dtype.csv')
dtype_dict = dtype_df.set_index('Column Name')['Data Type'].to_dict()
print("Loading benign packets datasets...")
benign_packets_df = pd.read_csv('benign_packets.csv', dtype=dtype_dict, low_memory=False)  # Load pre-sampled benign packets



# Final combined dataset 
print("Configuring Final Dataset...")
final_dataset = pd.concat([benign_packets_df, final_attack_df], ignore_index=True)
final_dataset.to_csv('semi-final-dataset.csv', index=False)
print(f"Combined dataset size: {len(final_dataset)}")
"""