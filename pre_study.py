import pandas as pd
import os
import glob

# Replace with your folder path
folder_path = '../../DDOS DataSet/CSV/Total'  # Specify your folder path here

# Constants
RANDOM_SEED = 42

# Gather all CSV files
all_files = glob.glob(os.path.join(folder_path, '*.csv'))

if not all_files:
    raise ValueError("No CSV files found in the specified folder.")

# Initialize counters and data holders
benign_count_total = 0
attack_sample_per_file = None
attack_samples_list = []

# First pass: count total benign packets
for file_path in all_files:
    for chunk in pd.read_csv(file_path, chunksize=10**6, low_memory=False):
        print(f"Processing file: {file_path}")
        benign_packets = chunk[chunk['Label'] == 'BENIGN']
        benign_count_total += len(benign_packets)

print(f"Total benign packets: {benign_count_total}")

# Set attack sample size
file_count = len(all_files)
attack_sample_total = 2 * benign_count_total
attack_sample_per_file = attack_sample_total // file_count

print(f"Attack samples per file: {attack_sample_per_file} on {file_count}")

# Collect attack samples
benign_packets_chunks = []

for file_path in all_files:
    attack_packets_from_file = []
    for chunk in pd.read_csv(file_path, chunksize=10**6, low_memory=False):
        print(f"Processing file: {file_path}")

        # Shuffle chunk before sampling
        chunk_shuffled = chunk.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

        # Collect benign packets
        benign_packets = chunk_shuffled[chunk_shuffled['Label'] == 'BENIGN']
        benign_packets_chunks.append(benign_packets)

        # Process attack packets
        attack_packets = chunk_shuffled[chunk_shuffled['Label'] != 'BENIGN']
        n_samples = min(len(attack_packets), attack_sample_per_file)
        sampled_attack = attack_packets.sample(n=n_samples, random_state=RANDOM_SEED)
        attack_packets_from_file.append(sampled_attack)

    attack_samples_list.extend(attack_packets_from_file)

# Combine all benign and attack packets
benign_packets_df = pd.concat(benign_packets_chunks, ignore_index=True)
attack_samples_df = pd.concat(attack_samples_list, ignore_index=True)

print("/nBenign Packets Samples:", len(benign_packets_df))
print("Attack Samples Collected:", len(attack_samples_df))
print("/n")

# Classify attack types
attack_types = attack_samples_df['Label'].unique()
attack_type_count = len(attack_types)
benign_per_attack_type = benign_count_total // attack_type_count

print(f"Attack types found: {attack_types}")

# Balance attack data across types
final_attack_samples = []

for attack_type in attack_types:
    attack_type_packets = attack_samples_df[attack_samples_df['Label'] == attack_type]
    n_samples = min(len(attack_type_packets), benign_per_attack_type)
    # Shuffle attack packets before sampling
    attack_type_packets_shuffled = attack_type_packets.sample(frac=1, random_state=RANDOM_SEED)
    sampled_packets = attack_type_packets_shuffled.head(n=n_samples)
    final_attack_samples.append(sampled_packets)

# Concatenate final attack samples
final_attack_df = pd.concat(final_attack_samples, ignore_index=True)
print(f"Final attack samples size: {len(final_attack_df)}")

# Shuffle benign and attack datasets again before final combine
# benign_shuffled = benign_packets_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
# attack_shuffled = final_attack_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# Final combined dataset 
final_dataset = pd.concat([benign_packets_df, final_attack_df], ignore_index=True)
print(f"Combined dataset size: {len(final_dataset)}")
final_dataset_shuffled = final_dataset.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"Final dataset size: {len(final_dataset)}")

# Export to CSV
final_dataset.to_csv('sampled_dataset.csv', index=False)

print("Sampling complete. Output saved to 'sampled_dataset.csv'.")