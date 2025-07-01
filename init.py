import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Load data from all CSV files in the specified folder
folder_path = '../../DDOS DataSet/CSV/Total'  # Specify your folder path here
# folder_path = './'
all_data = []

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        print(f"Loading file: {file_path}")
        data = pd.read_csv(file_path, low_memory=False)
        all_data.append(data)

# Concatenate all data into a single DataFrame
melbourne_data = pd.concat(all_data, ignore_index=True)

# Filter rows with missing values
melbourne_data = melbourne_data.dropna(axis=0)

# Map labels to "Benign" and "Attacking"
# melbourne_data['Label'] = melbourne_data['Label'].apply(lambda x: 'Benign' if x == 'BENIGN' else 'Attacking')

# Separate the benign and attacking packets
benign_data = melbourne_data[melbourne_data['Label'] == 'BENIGN']
attacking_data = melbourne_data[melbourne_data['Label'] != 'BENIGN']

# Determine the number of samples to keep for each class
n_benign = len(benign_data)
n_attacking = len(attacking_data)


# # Ensure equal distribution of attacking samples across all files
# # Calculate the number of attacking samples per file
# unique_attack_types = attacking_data['Label'].unique()
# print("\n\nUnique Attack Types:", unique_attack_types)
# attacks_per_file = n_attacking // len(unique_attack_types)

# Create a DataFrame to hold the balanced dataset
balanced_data = pd.DataFrame()

# Check if benign data is less than attacking data
if n_benign < n_attacking:
    # Downsample attacking packets to match benign data size
    attacks_per_file = n_benign // len(attacking_data['Label'].unique())
    print("\n Attacking data:", attacks_per_file)
    
    for attack in attacking_data['Label'].unique():
        attack_samples = attacking_data[attacking_data['Label'] == attack]
        attack_samples_downsampled = attack_samples.sample(n=attacks_per_file, random_state=1)
        print(f"Downsampled {len(attack_samples_downsampled)} samples for attack type: {attack}")
        balanced_data = pd.concat([balanced_data, attack_samples_downsampled])
    
    # Add all benign data to the balanced dataset
    balanced_data = pd.concat([balanced_data, benign_data])

else:
    # If benign data is greater than or equal to attacking data, downsample benign data
    benign_data_downsampled = benign_data.sample(n=n_attacking, random_state=1)
    balanced_data = pd.concat([benign_data_downsampled, attacking_data])

# Reset index of the balanced dataset
balanced_data.reset_index(drop=True, inplace=True)


# Check for infinite values and replace them if necessary
balanced_data.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
balanced_data.dropna(inplace=True)

# Shuffle the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=1).reset_index(drop=True)

# Split the balanced dataset into features and target
y = balanced_data['Label']
X = balanced_data[['Flow Bytes/s', 'Flow Packets/s', 'SYN Flag Count', 'Total Fwd Packets', 
                   'Total Backward Packets', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 
                   'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Duration', 
                   'RST Flag Count', 'Down/Up Ratio']]

# Split the data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Print the sizes of the new balanced datasets
print("Number of Benign packets in training set:", (train_y == 'BENIGN').sum())
print("Number of Attacking packets in training set:", (train_y != 'BENIGN').sum())

# Train the Random Forest model
forest_model = RandomForestClassifier(random_state=1)
forest_model.fit(train_X, train_y)

# Make predictions and evaluate the model
melb_preds = forest_model.predict(val_X)
accuracy = accuracy_score(val_y, melb_preds)
conf_matrix = confusion_matrix(val_y, melb_preds)
class_report = classification_report(val_y, melb_preds)

# Print results
print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
