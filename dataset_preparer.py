import numpy as np
import random
import os
import joblib
import scipy.io as sio
import datetime
from glob import glob
from utils import VerbosePrinter, setup_logger
import json
import matplotlib.pyplot as plt


def generate_selection_table(data_dir, subjs=5, sessions=2, MVC = [25], muscles = ["VL", "VM"], WS=40, SGs=3, ST=20, ext=".mat"):
    #search for all mat files recursively in the data_dir
    selections = {}
    for subj in range(1, subjs+1):
        for session in range(1, sessions+1):
            for ints in MVC:
                for mus in muscles:
                    subjDataName = f"Train_d{session}_uc{subj}_{ints}_{mus}"
                    selections[subjDataName] = []
                    for seg in range(1, SGs+1):
                        fnamePerSeg = f"Train_d{session}_uc{subj}_{ints}_{mus}-SG{seg}-WS{WS}-ST{ST}.mat"
                        if os.path.exists(os.path.join(data_dir, fnamePerSeg)):
                            selections[subjDataName].append(os.path.join(data_dir, fnamePerSeg))
                        else:
                            raise FileNotFoundError(f"File {fnamePerSeg} not found")
    return selections

def spikes2cst(spikes:np.ndarray):
    """
    Sum the spikes to get the CST.
    Args:
        spikes: numpy array of shape (N, num_outputs)
    Returns:
        cst_padded: numpy array of shape (N, 1)
    """
    cst = np.sum(spikes, axis=1)
    cst_padded = cst.reshape(-1, 1)
    return cst_padded

def encode_cst(raw_cst, num_outputs=4, plot=False, plot_path=None):
    """
    Decode a (N, 1) array into a (N, num_outputs) array based on the following rules:
    0 -> [0, 0, 0, 0]
    1 -> [1, 0, 0, 0]
    2 -> [1, 1, 0, 0]
    3 -> [1, 1, 1, 0]
    4+ -> [1, 1, 1, 1]
    
    Args:
        input_array: numpy array of shape (N, 1)
        num_columns: number of columns in output array (default 4)
    
    Returns:
        decoded_array: numpy array of shape (N, 4)
    """
    # Ensure input array is the right shape
    def plot_encoded_cst(raw_cst, encoded_cst, plot_path):
        plt.figure(figsize=(10, 5))
        plt.plot(raw_cst, label='Raw CST')
        encoded_cst = np.sum(encoded_cst, axis=1)
        plt.plot(encoded_cst, label='Encoded CST', alpha=0.5)
        plt.legend()
        plt.title(f"The encoded CST for the raw CST")
        plt.xlabel("Number of samples")
        plt.ylabel("CST")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()   

    if len(raw_cst.shape) == 1:
        raw_cst = raw_cst.reshape(-1, 1)
    
    N = raw_cst.shape[0]
    encoded_cst = np.zeros((N, num_outputs), dtype=int)
    
    # Clip values greater than 4 to 4
    clipped_values = np.clip(raw_cst, 0, num_outputs)
    
    # For each row, set the first n elements to 1 based on the input value
    for i in range(N):
        if clipped_values[i] > 0:
            # Extract single element using item() before converting to int
            num_ones = int(clipped_values[i].item())
            encoded_cst[i, :num_ones] = 1
    
    if plot and plot_path is not None:
        plot_encoded_cst(raw_cst, encoded_cst, plot_path)
            
    return encoded_cst


def make_random_selection(options:list,numbers_list:list[int], times:int, seed=42, saveDir=None, fname=None):
    options = np.asarray(options)
    rng = np.random.RandomState(seed)
    selected_dataset= np.array([])
    selected_dataset_history = []
    selection_record = {}
    for numbers in numbers_list:
        selection_record[f"mix_{numbers}"] = {}
        for j in range(1, times+1):
            selection_record[f"mix_{numbers}"][f"Selection{j}"] = {}
            selection_record[f"mix_{numbers}"][f"Selection{j}"]["selected_train_dataset"] = None
            selection_record[f"mix_{numbers}"][f"Selection{j}"]["leftover_dataset"] = None
            print(f"Selecting {numbers} datasets for the {j}th repetition")
            if seed is not None:
                rng = np.random.RandomState(seed)
            else:
                # Use current datetime as random seed
                seed = int(datetime.now().timestamp())
                rng = np.random.RandomState(seed)
            print(f"Using random seed: {seed}")
            current_selected_dataset = rng.choice(options, numbers, replace=False)
            selected_dataset = current_selected_dataset
            while tuple(current_selected_dataset) in [tuple(x) for x in selected_dataset_history]:
                print(f"Selected duplicate dataset, retrying...")
                current_selected_dataset = rng.choice(options, numbers, replace=False)
            selected_dataset = current_selected_dataset
            selected_dataset_history.append(current_selected_dataset)
            train_dataset_selected_list = selected_dataset.tolist()
            selection_record[f"mix_{numbers}"][f"Selection{j}"]["selected_train_dataset"] = train_dataset_selected_list
            leftover_dataset = np.setdiff1d(options, current_selected_dataset)
            selection_record[f"mix_{numbers}"][f"Selection{j}"]["leftover_dataset"] = leftover_dataset.tolist()
    if saveDir is not None:
        if fname is None:
            fname = f"Selection_{datetime.now().strftime('%Y:%m:%d-%H:%M:%S')}.json"
        
        with open(os.path.join(saveDir, fname), "w") as f:
            json.dump(selection_record, f)
    return selection_record


def load_dataset(matFile, raw_cst=True):
    vnames = ['EMGs', 'Spikes', 'Spikes_cst', 'CST_raw']
    data = sio.loadmat(matFile, variable_names=vnames)
    x_data = data["EMGs"]
    cst_raw = data["CST_raw"]
    x_data = x_data.astype(np.float32)
    if not raw_cst:
        labels = data["Spikes"]
        labels = spikes2cst(labels)
        labels = encode_cst(labels, num_outputs=4)
    else:
        labels = data["Spikes_cst"]
    return x_data, labels, cst_raw

def cnn_reshaper(emg, dim, electrodes=(13, 5)):
    if not dim in [1, 2, 3]:
        raise ValueError("Desired dimenison is not compatiable")
    if dim == 1:
        x_data = np.reshape(emg, (emg.shape[0], emg.shape[2], emg.shape[1]))
    elif dim == 2:
        emg_padded = np.pad(emg, ((0, 0), (0, 0), (1, 0)), 'constant')
        x_data = emg_padded.reshape(emg_padded.shape[0], emg_padded.shape[1], *electrodes)
    elif dim == 3:
        emg_padded = np.pad(emg, ((0, 0), (0, 0), (1, 0)), 'constant')
        x_data = emg_padded.reshape(emg_padded.shape[0], 1, emg_padded.shape[1], *electrodes)
    return x_data
        