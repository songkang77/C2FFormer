import os

import h5py
import numpy as np


def get_datasets_path_MAR(data_dir,name,rate):

    train_set_path = os.path.join(data_dir, "train.h5")
    val_set_path = os.path.join(data_dir, "val.h5")
    test_set_path = os.path.join(data_dir, "test.h5")

    with h5py.File(train_set_path, "r") as hf:
        if name == 'physionet_2012':
            train_X_ori = hf["X"][:]
        else:
            train_X_ori = hf["X_ori"][:]
    with h5py.File(val_set_path, "r") as hf:
        val_X_ori_arr = hf["X_ori"][:]
    prepared_train_set = train_X_ori
    prepared_val_ori_arr = val_X_ori_arr


    with h5py.File(test_set_path, "r") as hf:
        test_X_ori_arr = hf["X_ori"][:]  # need test_X_ori_arr to calculate MAE and MSE


    # test_indicating_arr = ~np.isnan(test_X_ori_arr) ^ ~np.isnan(test_X_arr)
    test_X_ori_arr = np.nan_to_num(test_X_ori_arr)
    if name =='beijing' or name =='pedestrian' or name == 'physionet_2012':
        prepared_train_set = fill_nan_with_nearest_vectorized(prepared_train_set)# 最近邻插值填上缺失值
        prepared_val_ori_arr = fill_nan_with_nearest_vectorized(prepared_val_ori_arr)
    if name != 'pedestrian':
        if rate =='_rate0.9':
            test_X_arr = create_fixed_interval_missing(test_X_ori_arr, missing_rate=0.9)
            val_X_arr = create_fixed_interval_missing(val_X_ori_arr, missing_rate=0.9)
        if rate =='_rate0.5':
            test_X_arr = create_fixed_interval_missing(test_X_ori_arr, missing_rate=0.5)
            val_X_arr = create_fixed_interval_missing(val_X_ori_arr, missing_rate=0.5)
        if rate =='_rate0.1':
            test_X_arr = create_fixed_interval_missing(test_X_ori_arr, missing_rate=0.1)
            val_X_arr = create_fixed_interval_missing(val_X_ori_arr, missing_rate=0.1)
        prepared_val_arr = val_X_arr
        test_indicating_arr = ~np.isnan(test_X_ori_arr) ^ ~np.isnan(test_X_arr)
    if name =='pedestrian':
        test_X_ori_arr = test_X_ori_arr.transpose(0, 2, 1).astype(np.float32)
        val_X_ori_arr = val_X_ori_arr.transpose(0, 2, 1).astype(np.float32)
        if rate =='_rate0.9':
            test_X_arr = create_fixed_interval_missing(test_X_ori_arr, missing_rate=0.9)
            val_X_arr = create_fixed_interval_missing(val_X_ori_arr, missing_rate=0.9)
        if rate =='_rate0.5':
            test_X_arr = create_fixed_interval_missing(test_X_ori_arr, missing_rate=0.5)
            val_X_arr = create_fixed_interval_missing(val_X_ori_arr, missing_rate=0.5)
        if rate =='_rate0.1':
            test_X_arr = create_fixed_interval_missing(test_X_ori_arr, missing_rate=0.1)
            val_X_arr = create_fixed_interval_missing(val_X_ori_arr, missing_rate=0.1)
        prepared_val_arr = val_X_arr
        test_indicating_arr = ~np.isnan(test_X_ori_arr) ^ ~np.isnan(test_X_arr)
        prepared_train_set = prepared_train_set.transpose(0, 2, 1).astype(np.float32)
        prepared_val_ori_arr = prepared_val_ori_arr.transpose(0, 2, 1).astype(np.float32)
    
    nan_count = np.isnan(prepared_train_set).sum()
    print(f"Number of NaNs in prepared_train_set: {nan_count}")
    nan_count_val = np.isnan(prepared_val_arr).sum()
    print(f"Number of NaNs in prepared_val_arr: {nan_count_val}")
    nan_count_test = np.isnan(test_X_arr).sum()
    print(f"Number of NaNs in test_X_arr: {nan_count_test}")    
    nan_count_test_ori = np.isnan(test_X_ori_arr).sum()
    print(f"Number of NaNs in test_X_ori_arr: {nan_count_test_ori}")
    true_count = test_indicating_arr.sum()
    print(f"Number of True in test_indicating_arr: {true_count}")



    return (
        prepared_train_set,
        prepared_val_arr,
        prepared_val_ori_arr,
        test_X_arr,
        test_X_ori_arr,
        test_indicating_arr,
    )
def get_datasets_path_onepiece(data_dir,name):

    data_dir_05 = data_dir.replace("09", "05")
    data_dir_01 = data_dir.replace("09", "01")
    test_set_path_05 = os.path.join(data_dir_05, "test.h5")
    test_set_path_01 = os.path.join(data_dir_01, "test.h5")
    with h5py.File(test_set_path_05, "r") as hf:
        test_X_arr_05 = hf["X"][:]
        test_X_ori_arr_05 = hf["X_ori"][:]  # need test_X_ori_arr to calculate MAE and MSE
    test_indicating_arr_05 = ~np.isnan(test_X_ori_arr_05) ^ ~np.isnan(test_X_arr_05)
    test_X_ori_arr_05 = np.nan_to_num(test_X_ori_arr_05)
    with h5py.File(test_set_path_01, "r") as hf:
        test_X_arr_01 = hf["X"][:]
        test_X_ori_arr_01 = hf["X_ori"][:]  # need test_X_ori_arr to calculate MAE and MSE
    test_indicating_arr_01 = ~np.isnan(test_X_ori_arr_01) ^ ~np.isnan(test_X_arr_01)
    test_X_ori_arr_01 = np.nan_to_num(test_X_ori_arr_01)
    if name =='pedestrian':
        test_X_arr_05 = test_X_arr_05.transpose(0, 2, 1).astype(np.float32)
        test_X_ori_arr_05 = test_X_ori_arr_05.transpose(0, 2, 1).astype(np.float32)
        test_indicating_arr_05 = test_indicating_arr_05.transpose(0, 2, 1)
        test_X_arr_01 = test_X_arr_01.transpose(0, 2, 1).astype(np.float32)
        test_X_ori_arr_01 = test_X_ori_arr_01.transpose(0, 2, 1).astype(np.float32)
        test_indicating_arr_01 = test_indicating_arr_01.transpose(0, 2, 1)
    return (
        test_X_arr_05,
        test_X_ori_arr_05,
        test_indicating_arr_05,
        test_X_arr_01,
        test_X_ori_arr_01,
        test_indicating_arr_01,
    )
def create_fixed_interval_missing(tensor, missing_rate=0.9):

    import numpy as np
    
    result = tensor.copy()

    if missing_rate < 0.5:
        
        k_missing = max(int(1 / missing_rate), 2)
    
        mask = np.ones_like(tensor, dtype=bool)
        

        mask[:, ::k_missing, :] = False 
    else:

        k = max(int(1 / (1 - missing_rate)), 2)
        

        mask = np.zeros_like(tensor, dtype=bool)
        

        mask[:, ::k, :] = True 
    

    result[~mask] = np.nan
    
    return result
def get_datasets_path(data_dir,name):
  
    train_set_path = os.path.join(data_dir, "train.h5")
    val_set_path = os.path.join(data_dir, "val.h5")
    test_set_path = os.path.join(data_dir, "test.h5")
    
    
    with h5py.File(train_set_path, "r") as hf:
        # train_X_arr = hf["X"][:]
        if name == 'physionet_2012':
            train_X_ori = hf["X"][:]
        else:
            train_X_ori = hf["X_ori"][:]
    with h5py.File(val_set_path, "r") as hf:
        val_X_arr = hf["X"][:]
        val_X_ori_arr = hf["X_ori"][:]
    prepared_train_set = train_X_ori
    print("prepared_train_set shape:", prepared_train_set.shape)
    prepared_val_arr = val_X_arr 
    prepared_val_ori_arr = val_X_ori_arr

    # nan_count_val = np.isnan(prepared_val_arr).sum()
    # print(f"Number of NaNs in prepared_val_arr: {nan_count_val}")
    with h5py.File(test_set_path, "r") as hf:
        test_X_arr = hf["X"][:]
        test_X_ori_arr = hf["X_ori"][:]  # need test_X_ori_arr to calculate MAE and MSE
    
    test_indicating_arr = ~np.isnan(test_X_ori_arr) ^ ~np.isnan(test_X_arr)
    
    test_X_ori_arr = np.nan_to_num(test_X_ori_arr)

    if name =='beijing' or name =='pedestrian' or name == 'physionet_2012':
        prepared_train_set = fill_nan_with_nearest_vectorized(prepared_train_set)
        prepared_val_ori_arr = fill_nan_with_nearest_vectorized(prepared_val_ori_arr)
    if name =='pedestrian':
        prepared_train_set = prepared_train_set.transpose(0, 2, 1).astype(np.float32)
        prepared_val_arr = prepared_val_arr.transpose(0, 2, 1).astype(np.float32)
        prepared_val_ori_arr = prepared_val_ori_arr.transpose(0, 2, 1).astype(np.float32)
        test_X_arr = test_X_arr.transpose(0, 2, 1).astype(np.float32)
        test_X_ori_arr = test_X_ori_arr.transpose(0, 2, 1).astype(np.float32)
        test_indicating_arr = test_indicating_arr.transpose(0, 2, 1)

    return (
        prepared_train_set,
        prepared_val_arr,
        prepared_val_ori_arr,
        test_X_arr,
        test_X_ori_arr,
        test_indicating_arr,
    )

def fill_nan_with_nearest_vectorized(arr):
    """
    Args:
        arr: (batch_size, sequence_length, n_features)
        
    Returns:
        filled_arr: 
    """
    filled_arr = arr.copy()
    
    for b in range(arr.shape[0]):
        for f in range(arr.shape[2]):
            seq = arr[b, :, f]
            mask = np.isnan(seq)
            
            if np.any(mask):
                # 
                valid_indices = np.where(~mask)[0]
                
                if len(valid_indices) > 0:
                    # 
                    idx_grid = np.arange(len(seq))
                    # 
                    distances = np.abs(idx_grid[:, np.newaxis] - valid_indices)
                    # 
                    nearest_idx = valid_indices[np.argmin(distances, axis=1)]
                    # 
                    filled_arr[b, mask, f] = arr[b, nearest_idx[mask], f]
                else:
                    filled_arr[b, :, f] = 0
                    
    return filled_arr

