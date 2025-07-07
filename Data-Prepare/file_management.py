import os
import glob
import pandas as pd
import numpy as np

MAIN_DATA_DIR = "D:\\Lucas\\Tensorprox\\DDOS DataSet\\CSV\\Total"
columns_to_train = ['Label', 'Init_Win_bytes_forward', 'Subflow Bwd Bytes', 'Subflow Bwd Packets', 'Subflow Fwd Bytes', 
                    'Packet Length Variance', 'Packet Length Std', 'Packet Length Mean', 'Min Packet Length', 'Bwd Packets/s', 
                    'Fwd Packets/s', 'Bwd Header Length', 'Bwd IAT Max', 'Bwd IAT Std', 'Bwd IAT Mean', 'Bwd IAT Total', 'Flow Bytes/s', 
                    'Fwd Packet Length Max', 'Total Length of Bwd Packets', 'Total Backward Packets', 'Fwd Packet Length Max', 
                    'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Mean', 'Bwd Packet Length Std']

def get_all_files():
    """
    Get all CSV files in the main data directory.
    """
    return glob.glob(os.path.join(MAIN_DATA_DIR, '*.csv'))


#  choose common columns in all files
def get_common_columns(files):
    """
    Get common columns across all CSV files.
    """
    common_columns = None
    for file in files:
        df = pd.read_csv(file, nrows=0)  # Read only the header
        if common_columns is None:
            common_columns = set(df.columns)
        else:
            common_columns.intersection_update(df.columns)
    return list(common_columns)


def save_common_columns_to_csv(output_file='common_columns.csv'):
    """
    Save the common columns to a CSV file.
    """
    common_columns = get_common_columns(get_all_files())
    # common_columns = np.array(common_columns).reshape(-1, 1).tolist()  # Reshape to a 2D list for better readability
    df = pd.DataFrame(common_columns, columns=['Column Name'])
    df.to_csv(output_file, index=False)
    # print(f"Common columns saved to {output_file}")

def load_common_columns_from_csv(input_file='common_columns.csv'):
    """
    Load common columns from a CSV file.
    """
    df = pd.read_csv(input_file)
    return df['Column Name'].tolist()

# test if the columns_to_train are in the common columns
def check_columns_in_common(columns_to_train, common_columns):
    """
    Check if the specified columns are in the common columns.
    """
    print("Checking if columns_to_train are in the common columns...")
    for col in columns_to_train:
        if col not in common_columns:
            print(f"Column {col} is not in the common columns.")
        else:
            print(f"Column {col} is in the common columns.")


# return filtered fiels that are ends_with _filtered.csv
def get_filtered_files():
    """
    Get all filtered CSV files in the main data directory.
    """
    return glob.glob(os.path.join(MAIN_DATA_DIR, '*_filtered.csv'))

def get_prepared_data_files():
    """
    Get all prepared data files in the prepared_data directory.
    """
    return glob.glob(os.path.join('prepared_data', '*.csv'))

def get_cleaned_data_files():
    """
    Get all cleaned data files in the cleaned_data directory.
    """
    return glob.glob(os.path.join('cleaned_data', '*.csv'))

# save_common_columns_to_csv()

# columns = load_common_columns_from_csv()

# columns_to_train = ['Label', 'Flow Bytes/s', 'Flow Packets/s', 'SYN Flag Count', 'Total Fwd Packets', 'Total Backward Packets', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 
#                 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Duration']
# print(len(columns_to_train), "columns to train")
# check_columns_in_common(columns_to_train, load_common_columns_from_csv())
# print(get_prepared_data_files)