import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
melbourne_file_path = './semi-final-dataset.csv'
dtype_df = pd.read_csv('semi_final_dtype.csv')
dtype_dict = dtype_df.set_index('Column Name')['Data Type'].to_dict()
print(f"Loading data from: {melbourne_file_path}")
melbourne_data = pd.read_csv(melbourne_file_path, dtype=dtype_dict)  # Load pre-sampled dataset

# Filter rows with missing values
# melbourne_data = melbourne_data.dropna(axis=0)
print(melbourne_data['Label'])

# Map labels to "Benign" and "Attacking"
print("Mapping labels to 'Benign' and 'Attacking'")
melbourne_data['Label'] = melbourne_data['Label'].str.strip('"').str.lower().apply(lambda x: 'Benign' if x == 'benign' else 'Attacking')

# Separate the benign and attacking packets
benign_data = melbourne_data[melbourne_data['Label'] == 'Benign']
attacking_data = melbourne_data[melbourne_data['Label'] == 'Attacking']

# Determine the number of samples to keep for each class
n_benign = len(benign_data)
n_attacking = len(attacking_data)

# Print the number of benign and attacking packets
print(f"Number of Benign packets: {n_benign}")
print(f"Number of Attacking packets: {n_attacking}")
balanced_data = pd.concat([benign_data, attacking_data])

# If benign packets are fewer, we will upsample them
# if n_benign < n_attacking:
#     # Upsample benign packets
#     benign_data_upsampled = benign_data.sample(n=n_attacking, replace=True, random_state=1)
#     balanced_data = pd.concat([benign_data_upsampled, attacking_data])
# else:
#     # Downsample attacking packets
#     attacking_data_downsampled = attacking_data.sample(n=n_benign, replace=True, random_state=1)
#     balanced_data = pd.concat([benign_data, attacking_data_downsampled])

# # Check for infinite values and replace them if necessary
balanced_data.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
balanced_data.dropna(inplace=True)

# # Shuffle the balanced dataset
# balanced_data = balanced_data.sample(frac=1, random_state=1).reset_index(drop=True)

# Now you can proceed to split the balanced dataset into features and target
y = balanced_data['Label']
X = balanced_data[['Flow Bytes/s', 'Flow Packets/s', 'SYN Flag Count', 'Total Fwd Packets', 'Total Backward Packets', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 
                'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Duration', 'RST Flag Count', 'Down/Up Ratio']]

# Split the data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0, shuffle=True)

# Print the sizes of the new balanced datasets
print("Number of Benign packets in training set:", (train_y == 'Benign').sum())
print("Number of Attacking packets in training set:", (train_X == 'Attacking').sum())

# Proceed to train your model as before
from sklearn.ensemble import RandomForestClassifier

forest_model = RandomForestClassifier(random_state=1)
forest_model.fit(train_X, train_y)

melb_preds = forest_model.predict(val_X)

# Make predictions and evaluate the model as before
accuracy = accuracy_score(val_y, melb_preds)
conf_matrix = confusion_matrix(val_y, melb_preds)
class_report = classification_report(val_y, melb_preds)

# # Print results
print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)