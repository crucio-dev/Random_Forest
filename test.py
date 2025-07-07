import os
import glob
import pandas as pd
import numpy as np

MAIN_DATA_DIR = "D:\\Lucas\\Tensorprox\\DDOS DataSet\\CSV\\Total"

# Get 300 rows from attacking_samples and benign_packets and save that to a new file
def get_benign_attacking_samples():
    datas = []
    for i in range(1, 6):
        benign_samples = pd.read_csv('benign_packets.csv', nrows=20)
        attacking_samples = pd.read_csv('attack_samples.csv', nrows=20)
        datas.append(benign_samples)
        datas.append(attacking_samples)
    
    combined_data = pd.concat(datas, ignore_index=True)
    combined_data.to_csv('benign_attacking_samples.csv', index=False)
        
        
    
    


get_benign_attacking_samples()