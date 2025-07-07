import pandas as pd
import glob
import gc
import os



def sample_collecting(RANDOM_SEED, all_files, benign_count_total, attack_sample_per_file):
    # Collect attack samples
    folder_path = '../../DDOS DataSet/CSV/Total'  # Specify your folder path here

    # RANDOM_SEED = 42  # Set a random seed for reproducibility
    # all_files = glob.glob(os.path.join(folder_path, '*.csv'))
    # benign_count_total = 7300017  # Total benign packets count
    # attack_sample_per_file = 2 * benign_count_total  # Total attack samples to collect
    attack_samples_list = []

    for file_path in all_files:
        print(f"Processing file: {file_path}")
        # if file_path.endswith('cic-collection.csv'):
        #     print("Skipping cic-collection file.")
        #     continue

        # Read file in chunks to limit memory usage
        chunk_list = []
        chunksize = 10 ** 6  # Adjust based on your system's memory
        reader = pd.read_csv(file_path, chunksize=chunksize, low_memory=False)

        for chunk in reader:
            # Filter out benign packets
            chunk_filtered = chunk[chunk['Label'].str.lower() != 'benign']

            # Shuffle only if sampling across entire file is essential
            # Skip shuffling here if not necessary
            # chunk_filtered = chunk_filtered.sample(frac=1, random_state=RANDOM_SEED)

            chunk_list.append(chunk_filtered)

        # Concatenate all filtered chunks at once (only if memory permits)
        packets = pd.concat(chunk_list, ignore_index=True)

        # Determine number of samples
        n_samples = min(len(packets), int(attack_sample_per_file))
        # Sample attack packets
        packets = packets.sample(n=n_samples, random_state=RANDOM_SEED)

        attack_samples_list.append(packets)

        # Free memory
        del packets
        gc.collect()

    # Combine all benign and attack packets
    attack_samples_df = pd.concat(attack_samples_list, ignore_index=True)

    print("Attack Samples Collected:", len(attack_samples_df))
    print("/n")

    attack_samples_list.clear()
    attack_samples_df.to_csv('attack_samples.csv', index=False)  # Save to CSV if needed
    print("Attack samples saved to 'attack_samples.csv'")   