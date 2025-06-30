# import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
# # Load data
# melbourne_file_path = './UDP.csv'
# melbourne_data = pd.read_csv(melbourne_file_path) 

# # Filter rows with missing values
# melbourne_data = melbourne_data.dropna(axis=0)

# # Map labels to "Benign" and "Attacking"
# melbourne_data['Label'] = melbourne_data['Label'].apply(lambda x: 'Benign' if x == 'Benign' else 'Attacking')

# # Choose target and features
# y = melbourne_data['Label']

# # melbourne_features = ['Source Port', 'Protocol', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
# #                         'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
# #                         'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
# #                         'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
# #                         'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
# #                         'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
# #                         'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
# #                         'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
# #                         'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
# #                         'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size',
# #                         'Avg Bwd Segment Size', 'Fwd Header Length.1',
# #                         'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk'
# #                     ]
# melbourne_features = ['Total Fwd Packets']
# X = melbourne_data[melbourne_features]

# # split data into training and validation data, for both features and target
# # The split is based on a random number generator. Supplying a numeric value to
# # the random_state argument guarantees we get the same split every time we
# # run this script.
# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# # print("Train X>>>: \n", train_X)
# # print("\nTrain Y>>>: \n", train_y) 

# forest_model = RandomForestClassifier(random_state=1)
# forest_model.fit(train_X, train_y)
# melb_preds = forest_model.predict(val_X)
# # print(melb_preds)
# # print("Results from Random Forest model:", melb_preds)
# accuracy = accuracy_score(val_y, melb_preds)
# conf_matrix = confusion_matrix(val_y, melb_preds)
# class_report = classification_report(val_y, melb_preds)

# # Print results
# print("Accuracy:", accuracy)
# print("\nConfusion Matrix:\n", conf_matrix)
# print("\nClassification Report:\n", class_report)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
melbourne_file_path = './UDP.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# Filter rows with missing values
melbourne_data = melbourne_data.dropna(axis=0)

# Map labels to "Benign" and "Attacking"
melbourne_data['Label'] = melbourne_data['Label'].apply(lambda x: 'Benign' if x == 'BENIGN' else 'Attacking')

# Separate the benign and attacking packets
benign_data = melbourne_data[melbourne_data['Label'] == 'Benign']
attacking_data = melbourne_data[melbourne_data['Label'] == 'Attacking']

# Determine the number of samples to keep for each class
n_benign = len(benign_data)
n_attacking = len(attacking_data)


# If benign packets are fewer, we will upsample them
if n_benign < n_attacking:
    # Upsample benign packets
    benign_data_upsampled = benign_data.sample(n=n_attacking, replace=True, random_state=1)
    balanced_data = pd.concat([benign_data_upsampled, attacking_data])
else:
    # Downsample attacking packets
    attacking_data_downsampled = attacking_data.sample(n=n_benign, random_state=1)
    balanced_data = pd.concat([benign_data, attacking_data_downsampled])

# Shuffle the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=1).reset_index(drop=True)

# Now you can proceed to split the balanced dataset into features and target
# y = balanced_data['Label']
# X = balanced_data[['Down/Up Ratio']]

# Split the data into training and validation sets
# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# # # Print the sizes of the new balanced datasets
# # print("Number of Benign packets in training set:", (train_y == 'Benign').sum())
# # print("Number of Attacking packets in training set:", (train_y == 'Attacking').sum())

# # Proceed to train your model as before
# from sklearn.ensemble import RandomForestClassifier

# forest_model = RandomForestClassifier(random_state=1)
# forest_model.fit(train_X, train_y)

# melb_preds = forest_model.predict(val_X)

# # Make predictions and evaluate the model as before
# accuracy = accuracy_score(val_y, melb_preds)
# conf_matrix = confusion_matrix(val_y, melb_preds)
# class_report = classification_report(val_y, melb_preds)

# # # Print results
# print("Accuracy:", accuracy)
# print("\nConfusion Matrix:\n", conf_matrix)
# print("\nClassification Report:\n", class_report)
