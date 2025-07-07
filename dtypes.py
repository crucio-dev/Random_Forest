import pandas as pd

# Load the CSV file
df_attack = pd.read_csv('semi-final-dataset.csv')

# Get the data types of each column
column_types = df_attack.dtypes

# Print the column types
print(column_types)

# Convert the Series to a DataFrame for proper CSV structure
column_types_df = column_types.reset_index()
column_types_df.columns = ['Column Name', 'Data Type']  # Rename columns

# Save the column types to a CSV file (without index)
column_types_df.to_csv('semi_final_dtype.csv', index=False)




