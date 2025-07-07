import file_management as fm
import os
import glob
import handle_pickle as hp
import pandas as pd
import numpy as np

files = fm.get_all_files()
common_columns = fm.columns_to_train
filtered_files = fm.get_filtered_files()
prepared_data_files = fm.get_prepared_data_files()
cleaned_data_files = fm.get_cleaned_data_files()

BENIGN_PACKET_COUNT = 5e6

# filter all files to only include common columns in every file and save to new files in ./prepared_data avoiding out of memory issues(the files are too big)
def filter_files_to_common_columns(files, common_columns):
    """
    Filter all files to only include common columns and save to new files.
    """
    for file in files:
        df = pd.read_csv(file, usecols=common_columns)
        output_file = file.replace('.csv', '_filtered.csv')
        df.to_csv(output_file, index=False)
        print(f"Filtered file saved as {output_file}")

# filter_files_to_common_columns(files, common_columns)


# move filtered files to prepared_data directory
def move_filtered_files_to_prepared_data(filtered_files, target_directory='prepared_data'):
    """
    Move filtered files to the specified target directory.
    """
    import os
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    for file in filtered_files:
        base_name = os.path.basename(file)
        target_path = os.path.join(target_directory, base_name)
        os.rename(file, target_path)
        print(f"Moved {file} to {target_path}")

    
# remove the rows that includes the 'NA', 'Inf' values from the filtered files
def remove_invalid_rows(prepared_data_files):
    """
    Remove rows with 'NA' or 'Inf' values from the filtered files.
    """
    for file in prepared_data_files:
        print(f"Processing file: {file}")
        df = pd.read_csv(file)
        initial_row_count = len(df)

        # Replace invalid values and drop NaNs
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        final_row_count = len(df)
        print(f"Removed {(1 - final_row_count / initial_row_count) * 100:.2f}% invalid rows from {file}")

        # Remove duplicates
        df = df.drop_duplicates()

        final_row_count = len(df)
        print(f"Removed {(1 - final_row_count / initial_row_count) * 100:.2f}% duplicated rows from {file}")

        # Get filename and save to cleaned_data folder
        filename = os.path.basename(file)
        save_file = filename.replace('_filtered.csv', '_cleaned.csv')
        save_path = os.path.join('cleaned_data', save_file)
        df.to_csv(save_path, index=False)

# get columns has NA and inf values and store that to a csv file with prepared data file name
def get_invalid_columns(prepared_data_files):
    """
    Get columns with 'NA' or 'Inf' values and save to a CSV file.
    """
    invalid_columns = {}
    
    for file in prepared_data_files:
        print(f"Processing file: {file}")
        df = pd.read_csv(file)
        na_columns = df.columns[df.isna().any()].tolist()
        inf_columns = df.columns[(df == np.inf).any()].tolist()
        
        if na_columns or inf_columns:
            invalid_columns[file] = {
                'File': file,
                'NA Columns': na_columns,
                'Inf Columns': inf_columns
            }
    
    # Save the invalid columns to a CSV file
    invalid_columns_df = pd.DataFrame.from_dict(invalid_columns, orient='index')
    invalid_columns_df.to_csv('invalid_columns.csv')
    print("Invalid columns saved to invalid_columns.csv")

# get the total count of packet types from 'Label' column in each filtered file and save that to the csv file
def get_packet_type_counts(cleaned_data_files):
    """
    Get the count of each packet type from the 'Label' column in each cleaned file.
    """
    packet_type_counts = {}
    
    for file in cleaned_data_files:
        print(f"Processing file: {file}")
        df = pd.read_csv(file)
        counts = df['Label'].value_counts().to_dict()
        packet_type_counts[file] = counts
    
    # Save the packet type counts to a CSV file
    packet_type_counts_df = pd.DataFrame.from_dict(packet_type_counts, orient='index').fillna(0).astype(int)
    packet_type_counts_df.to_csv('packet_type_counts.csv')
    print("Packet type counts saved to packet_type_counts.csv")


# print(len(filtered_files), "filtered files found.")
# for file in filtered_files:
#     print(f"Processing file: {file}")

# move_filtered_files_to_prepared_data(filtered_files)
# filter_files_to_common_columns(files, common_columns)
# remove_invalid_rows(prepared_data_files)
# get_invalid_columns(prepared_data_files)
get_packet_type_counts(cleaned_data_files)