import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import VerbosePrinter
import os
from copy import deepcopy
from torchmetrics.classification import BinaryF1Score
import scipy
from scipy import signal
from typing import Optional, Tuple, Callable
import matplotlib.pyplot as plt

def rescale(x, new_min=0.0, new_max=30.0, old_min=None, old_max=None):
    x = np.asarray(x, dtype=float)
    if old_min is None: old_min = x.min()
    if old_max is None: old_max = x.max()
    return (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def smooth_cst_hanning_convolve(x, hws, fs, ss, convolve_mode="same"):
    """
    Smooth the CST signal using a Hanning window
    Args:
        x: the CST signal to be smoothed
    Returns:
        smoothed_cst: the smoothed CST signal
        index: the index of the smoothed CST signal
    """
    time_period = hws / 1000
    n_samples_float = int((fs * time_period/ss))
    window = signal.windows.hann(n_samples_float)
    window = window / window.sum()
    smoothed_cst = scipy.signal.convolve(x, window, mode=convolve_mode)
    index = np.arange(len(smoothed_cst))
    return smoothed_cst, index

def process_neural_drive(cst_bss:np.ndarray, cst_cnn:np.ndarray):
    hws = 400
    fs = 2048
    ss = 20
    """
    This function scale the cst to 0 to mvc, and calculate the R and RMSE
    """
    smooth_cst_bss, index = smooth_cst_hanning_convolve(cst_bss, hws, fs, ss, "same")
    smooth_cst_cnn, index = smooth_cst_hanning_convolve(cst_cnn, hws, fs, ss, "same")
    corr_coeff = np.corrcoef(smooth_cst_bss, smooth_cst_cnn)[0, 1]
    rmse = np.sqrt(np.mean((smooth_cst_cnn - smooth_cst_bss) ** 2))
    nrmse = rmse / np.max(smooth_cst_bss)
    return corr_coeff, rmse, nrmse, smooth_cst_bss, smooth_cst_cnn



def save_checkpoint(model, epoch, optimizer, saveDir, namePrefix, best_metric=0, best_loss=float('inf') ):
    # check existing checkpoint file with same namePrefix and delete it
    for file in os.listdir(saveDir):
        if file.startswith(namePrefix) and file.endswith("_ckpt.pt"):
            os.remove(os.path.join(saveDir, file))
    checkpoint_path = os.path.join(saveDir, namePrefix+f"_{epoch}_ckpt.pt")
    model2save = deepcopy(model)
    model2save.to("cpu")
    # check if the checkpoint directory exists, if so delete it
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model2save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "best_loss": best_loss
        },
        checkpoint_path
    )
    del model2save

def check_early_stopping(val_score, train_loss, best_val_score, best_train_loss, patience, early_stop_counter, epoch, min_delta=0.001):
    early_stop = False
    best_model = False
    if best_val_score is None or val_score > best_val_score + min_delta:
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_val_score = val_score
            early_stop_counter = 0
            best_model = True
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            early_stop = True
            early_stop_counter = patience
    return early_stop, best_model, early_stop_counter, best_train_loss, best_val_score

def save_best_model(model, epoch, best_metric, best_loss, saveDir, namePrefix):
    checkpoint_path = os.path.join(saveDir, namePrefix+f"_best.pt")
    model2save = deepcopy(model)
    model2save.to("cpu")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    torch.save(
        {   
            "epoch": epoch,
            "model_state_dict": model2save.state_dict(),
            "best_metric": best_metric,
            "best_loss": best_loss
        },
        checkpoint_path
    )
    del model2save


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = None
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_model = False
    
    def __call__(self, val_score, train_loss, epoch):
        if self.best_metric is None or (val_score > self.best_metric + self.min_delta and train_loss < self.best_loss):
            self.best_loss = train_loss
            self.best_metric = val_score
            self.counter = 0
            self.best_model = True
            self.early_stop = False
        else:
            self.counter += 1
            self.best_model = False
            self.early_stop = False
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop, self.best_model, self.best_loss, self.best_metric
                


class NeuralDataset(Dataset):
    def __init__(
        self, 
        data: np.ndarray, 
        labels: np.ndarray, 
        transform: Optional[callable] = None
    ) -> None:
        # Convert numpy arrays to torch tensors and ensure float type
        if data.dtype != np.float32:
            raise ValueError(f"The data type of the data is {data.dtype}, it should be float32, please convert the data to float32")
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.transform = transform
        
        # Validate input dimensions
        assert len(self.data) == len(self.labels), "Data and labels must have same length"

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx]
        y = self.labels[idx]
        
        if self.transform is not None:
            x = self.transform(x)
            
        return x, y


def inference_model(model, inference_dataloader:DataLoader, device:torch.device):
    model = model.to(device)
    model.eval()
    evaluator = BinaryF1Score(multidim_average="global").to(device)
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for x, y in inference_dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            all_outputs.append(output)
            all_labels.append(y)
    outputs = torch.cat(all_outputs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    f1 = evaluator(outputs, labels).cpu().numpy()
    accuracy = (outputs.round() == labels).float().mean().item()
    pred_cst = torch.sum(outputs.round(), dim=1).cpu().numpy()
    bss_cst = torch.sum(labels, dim=1).cpu().numpy()
    corr_coeff, rmse, nrmse, smooth_cst_bss, smooth_cst_cnn = process_neural_drive(bss_cst, pred_cst)
    return f1, accuracy, corr_coeff, rmse, nrmse, pred_cst, bss_cst, smooth_cst_bss, smooth_cst_cnn

def inference_model_simple_version(model, inference_dataloader:DataLoader, device:torch.device):
    model = model.to(device)
    model.eval()
    evaluator = BinaryF1Score(multidim_average="global").to(device)
    all_outputs = []
    all_labels = []
    with torch.inference_mode():
        for x, y in inference_dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            all_outputs.append(output)
            all_labels.append(y)
        outputs = torch.cat(all_outputs, dim=0)
        labels = torch.cat(all_labels, dim=0)
        f1 = evaluator(outputs, labels).cpu().numpy()
        accuracy = (outputs.round() == labels).float().mean().item()
        pred_cst = torch.sum(outputs.round(), dim=1).cpu().numpy()
        bss_cst = torch.sum(labels, dim=1).cpu().numpy()
    return pred_cst, bss_cst, f1, accuracy
def plot_train_history(train_result, save_dir, namePrefix, variable):
    plt.figure(figsize=(10, 8))
    plt.plot(train_result, label=variable)
    plt.xlabel("Epoch")
    plt.ylabel(variable)
    plt.title(f"Training History of {variable} for {namePrefix}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{namePrefix}_{variable}.png"))
    plt.close()
