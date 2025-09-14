# Libraries required
from scipy.io import arff
import numpy as np
import pandas as pd
import os
# import pickle
# from Utils.utils import seed_everything

data_path = None

def set_data_path_(path):
    global data_path
    data_path = path

def data_folder_path(data_folder, data_name):
    if data_path is None:
        raise ValueError("`data_path` is set to None. Set `data_path` to the path of the 'Data' folder containing all the datasets.\n\
        Use `data_load.set_data_path()` method inside the DataCode module to set the path")
    return os.path.join(os.path.abspath(data_path), data_folder, data_name)

def gen_colors(n, seed=37):
    np.random.seed(seed)
    
    def generate_colors(count):
        # Generate colors as integers between 0 and 255
        return np.random.randint(0, 256, size=(count, 3))
    
    # Generate initial set of colors
    colors_int = generate_colors(n)
    
    # Check for uniqueness and generate more if needed
    unique_colors_int = np.unique(colors_int, axis=0)
    while len(unique_colors_int) < n:
        additional_colors = generate_colors(n - len(unique_colors_int))
        colors_int = np.vstack((unique_colors_int, additional_colors))
        unique_colors_int = np.unique(colors_int, axis=0)
    
    # Convert to float and round to 3 decimal places
    unique_colors_float = np.round(unique_colors_int.astype(float) / 255, 3)
    
    return unique_colors_float[:n]

# functions to load various datasets

# Load dry bean data
def data_load_dry_bean(data_folder):
    data_name = "Dry_Bean_Dataset.arff"
    data_path = data_folder_path(data_folder, data_name)
    data, meta = arff.loadarff(data_path)

    data_initial  = pd.DataFrame(data)
    data_initial['Class'] = data_initial['Class'].str.decode('utf-8')

    class_encoding = {
        'SEKER'   : 0,
        'BARBUNYA': 1,
        'BOMBAY'  : 2,
        'CALI'    : 3,
        'HOROZ'   : 4,
        'SIRA'    : 5,
        'DERMASON': 6
    }
    data_initial['Class'] = data_initial['Class'].map(class_encoding)

    data = data_initial.sample(frac=1, random_state=42)

    X = data.drop(columns=["Class"]).to_numpy()
    Y = data["Class"].to_numpy().reshape(-1, 1)
    colors = gen_colors(X.shape[1], seed=42)

    return X, Y, colors

# Load gas data
def data_load_gas(data_folder):
    X, Y = [], []

    # Load each batch file
    for i in range(1, 11):
        filepath = data_folder_path(data_folder, f"batch{i}.dat")
        with open(filepath, 'r') as f:
            for line in f:
                # Split label; format: label;concentration feat1 feat2 ... feat128
                parts = line.strip().split()
                label = int(parts[0].split(';')[0])-1  # get class number
                Y.append(label)
                
                # Extract features from the remaining x:v format
                features = [float(p.split(':')[1]) for p in parts[1:]]
                if len(features) != 128:
                    raise ValueError(f"Feature length mismatch in file {filepath}: got {len(features)} features.")
                X.append(features)
    
    X, Y = np.array(X, dtype=np.float32), np.array(Y, dtype=np.int32)
    colors = gen_colors(X.shape[1], seed=42)

    return X, Y, colors