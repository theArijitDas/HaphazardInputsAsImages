import numpy as np
from DataCode.data_utils import data_load_dry_bean, data_load_gas
from DataCode.data_utils import set_data_path_

def check_mask_each_instance(mask):
    index_0 = np.where(np.sum(mask, axis = 1) == 0)
    random_index = np.random.randint(mask.shape[1], size = (len(index_0[0])))
    # print(mask.shape, index_0, len(index_0[0]))
    for i in range(len(index_0[0])):
        mask[index_0[0][i], random_index[i]] = 1
    return mask

def set_data_path(path):
    set_data_path_(path)

def dataloader(data_folder='dry_bean',p_available=0.5,seed=42):
    # load data and unique colors using functions from data_utils.py
    if data_folder == 'dry_bean':
        X, Y, colors = data_load_dry_bean(data_folder)
    elif data_folder == 'gas':
        X, Y, color = data_load_gas(data_folder)
    num_inst=X.shape[0]
    num_feats=X.shape[1]
    num_classes=np.unique(Y).shape[0]

    np.random.seed(seed)
    mask = (np.random.random((num_inst, num_feats)) < p_available).astype(float)# randomly(seeded) drop values according to p_available. 
    # 0 means the feature is visible and 1 means the feature is missing
    rev_mask =(1-mask) # reverse mask for plotting nan values in the form of crosses, if needed.
    mask = check_mask_each_instance(mask) # check if any instance has all features missing, if so, randomly select one feature to be visible
    X_haphazard = np.where(mask, X, np.nan) # apply mask to data
    mat_rev_mask=np.where(rev_mask, 0.5, np.nan) # reverse mask for plotting nan values in the form of crosses, if needed.

    return X_haphazard, Y, num_classes, mat_rev_mask, colors
        
        
