# merge all csv files in dir 'cleaned_data' to a single csv file classified by Label after processing each Label in lowercase
import os
import pandas as pd
import glob
import handle_pickle as hp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import numpy as np
import file_management as fm
import data_preparation as dp
import warnings
warnings.filterwarnings("ignore")
# Define the columns to train the model
columns_to_train = fm.columns_to_train
# Get all cleaned data files
cleaned_data_files = fm.get_cleaned_data_files()
# merge all cleaned data files into a single DataFrame
def merge_cleaned_data_files(files):
    """
    Merge all cleaned data files into a single DataFrame.
    """
    df_list = []
    for file in files:
        df = pd.read_csv(file, usecols=columns_to_train)
        df_list.append(df)
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna()
    return merged_df

def prepare_data_for_training(df):
    """
    Prepare the data for training by encoding categorical variables and handling missing values.
    """
    # make uppercase and lowercase columns as one lowercased column
    df.columns = [col.lower() for col in df.columns]
    # Convert all columns to lowercase
    df.columns = df.columns.str.lower()
    # leave only the columns that are lowercased
    global columns_to_train
    columns_to_train = [col.lower() for col in columns_to_train]
    df = df[[col for col in df.columns if col in columns_to_train]]

    # Encode categorical variables    
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    df[df.select_dtypes(include=[np.number]).columns] = imputer.fit_transform(df.select_dtypes(include=[np.number]))

    return df, label_encoders

def train_random_forest_classifier(df):
    """
    Train a Random Forest Classifier on the prepared DataFrame.
    """
    # Split the data into features and target variable
    X = df.drop('label', axis=1)
    y = df['label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, verbose=3)

    # Fit the model
    rf_classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_classifier.predict(X_test)
    # print all classe types
    print("Classes in the target variable:", np.unique(y))
    # Evaluate the model
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    return rf_classifier

def save_model(model, filename='rf_classifier.pkl'):
    """
    Save the trained model to a file.
    """
    hp.save_variable_as_pickle(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename='rf_classifier.pkl'):
    """
    Load the trained model from a file.
    """
    model = hp.load_variable_from_pickle(filename)
    print(f"Model loaded from {filename}")
    return model

def main():
    # Merge cleaned data files
    merged_df = merge_cleaned_data_files(cleaned_data_files)
    print(f"Merged DataFrame shape: {merged_df.shape}")
    filered_encoders = ['BENIGN', 'DrDoS_DNS', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NTP',
       'DrDoS_NetBIOS', 'DrDoS_SNMP', 'DrDoS_SSDP', 'DrDoS_UDP', 'LDAP',
       'MSSQL', 'NetBIOS', 'Syn', 'TFTP', 'UDP', 'UDP-lag']
    merged_df = merged_df[merged_df['Label'].isin(filered_encoders)]
    # Prepare the data for training
    prepared_df, label_encoders = prepare_data_for_training(merged_df)
    print("label_encoders:", label_encoders.keys())
    print("all classes in label encoders:", {col: le.classes_ for col, le in label_encoders.items()})
    # exit(0)
    print(f"Prepared DataFrame shape: {prepared_df.shape}")

    # Train the Random Forest Classifier
    rf_classifier = train_random_forest_classifier(prepared_df)

    # Save the trained model
    save_model(rf_classifier)

    # Save the label encoders
    hp.save_variable_as_pickle(label_encoders, 'label_encoders.pkl')


if __name__ == "__main__":
    main()
    # Load the model for testing
    rf_classifier = load_model()
    # Load the label encoders for testing
    label_encoders = hp.load_pickle('label_encoders.pkl')
    print("Label encoders loaded successfully.")    
    