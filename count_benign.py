import pandas as pd
import os
import glob
import gc

# def count_benign_packets():
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
benign_packets_chunks = []

# First pass: count total benign packets
for file_path in all_files:
    for chunk in pd.read_csv(file_path, chunksize=10**6, low_memory=False):
        print(f"Processing file: {file_path}")
        benign_packets = chunk[chunk['Label'].str.strip('"').str.lower() == 'benign']
        benign_packets_chunks.append(benign_packets)

        del chunk, benign_packets  # Free memory
        gc.collect()  # Force garbage collection to free memory

benign_packets_df = pd.concat(benign_packets_chunks, ignore_index=True)
benign_count_total = len(benign_packets_df)
benign_packets_df.to_csv('benign_packets.csv', index=False)

# return all_files, benign_count_total, benign_packets_df