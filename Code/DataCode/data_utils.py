import numpy as np # type: ignore
import pandas as pd # type: ignore
import os
import pickle
from Utils.utils import seed_everything

def data_folder_path(data_folder, data_name):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Data', data_folder, data_name)

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

def data_load_magic04(data_folder):
    data_name = "magic04.data"
    data_path = data_folder_path(data_folder, data_name)
    print(data_path)
    data_initial =  pd.read_csv(data_path, sep = "," , header = None, engine = 'python')
    label = np.array(data_initial[10] == 'g')*1
    data_initial = data_initial.iloc[:,:10]
    data_initial.insert(0, column="class", value=label)
    data = data_initial.sample(frac = 1)

    Y = np.array(data.iloc[:,:1])
    X = np.array(data.iloc[:,1:])

    colors= gen_colors(X.shape[1], 42)

    return X, Y, colors

def data_load_a8a(data_folder):
    data_name = "a8a.txt"
    n_feat = 123
    number_of_instances = 32561
    data_path = data_folder_path(data_folder, data_name)
    data_initial =  pd.read_csv(data_path, sep = " ", header = None, engine = 'python')
    data = pd.DataFrame(0, index=range(number_of_instances), columns = list(range(1, n_feat+1)))
    # 16th column contains only NaN value
    data_initial = data_initial.iloc[:, :15]
    for j in range(data_initial.shape[0]):
            l = [int(i.split(":")[0])-1 for i in list(data_initial.iloc[j, 1:]) if not pd.isnull(i)]
            data.iloc[j, l] = 1
    label = np.array(data_initial[0] == -1)*1
    data.insert(0, column='class', value=label)
    data = data.sample(frac = 1)

    Y = np.array(data.iloc[:,:1])
    X = np.array(data.iloc[:,1:])

    colors= gen_colors(n_feat, 42)
    return X, Y, colors

def data_load_susy(data_folder):
    data_name = "SUSY_1M.csv.gz"
    data_path = data_folder_path(data_folder, data_name)
    data_initial =  pd.read_csv(data_path, compression='gzip')
    label = np.array(data_initial["0"] == 1.0)*1
    data_initial = data_initial.iloc[:,1:]
    data_initial.insert(0, column="class", value=label)
    data = data_initial.sample(frac = 1)

    Y = np.array(data.iloc[:,:1])
    X = np.array(data.iloc[:,1:])
    colors=gen_colors(X.shape[1], 42)
    return X, Y, colors

def data_load_higgs():
    #data_name = ""
    data_path = "/datasets/HIGGS_1M.csv.gz"
    data_initial =  pd.read_csv(data_path, compression='gzip')
    label = np.array(data_initial["0"] == 1.0)*1
    data_initial = data_initial.iloc[:,1:]
    data_initial.insert(0, column="class", value=label)
    data = data_initial.sample(frac = 1)

    Y = np.array(data.iloc[:,:1])
    X = np.array(data.iloc[:,1:])
    colors=gen_colors(X.shape[1], 42)

    return X, Y, colors