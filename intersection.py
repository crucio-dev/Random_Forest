"""
This script is for configuring the CSVs so that they have same columns and data types.
"""

import pandas as pd
import os

benign_dtypes = pd.read_csv('benign_packets_dtype.csv')
attack_dtypes = pd.read_csv('attack_samples_dtype.csv')

benign_columns = benign_dtypes['Column Name'].tolist()  # Adjust the column name as needed
attack_columns = attack_dtypes['Column Name'].tolist()  # Adjust the column name as needed

common_columns = list(set(benign_columns).intersection(set(attack_columns)))

print(f"Common columns: {common_columns}")


print("Loading benign packets dataset...")
dtype_df = pd.read_csv('benign_packets_dtype.csv')
dtype_dict = dtype_df.set_index('Column Name')['Data Type'].to_dict()
benign_df = pd.read_csv('benign_packets.csv', dtype=dtype_dict)  # Load pre-sampled benign packets

print("Loading attacking packets dataset...")
dtype_df = pd.read_csv('attack_samples_dtype.csv')
dtype_dict = dtype_df.set_index('Column Name')['Data Type'].to_dict()
attack_df = pd.read_csv('final_attack_samples.csv', dtype=dtype_dict)

benign_df = benign_df[common_columns]
attack_df = attack_df[common_columns]

print(f"Configuring the Final Dataset")
final_sample = pd.concat([benign_df, attack_df], ignore_index=True)

print(f"Combined dataset size: {len(final_sample)}")
print("Sampled Dataset.csv'...")
final_sample.to_csv('semi-final-dataset.csv', index=False)
