import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import VerbosePrinter
import os
from copy import deepcopy
from torchmetrics.classification import BinaryF1Score
from decoder_trainer_tools import check_early_stopping
from decoder_trainer_tools import save_checkpoint
from decoder_trainer_tools import save_best_model
from decoder_trainer_tools import EarlyStopping


def decoder_trainer_validator(
        model, 
        trainLoader, 
        valLoader, 
        criterion, 
        optimizer, 
        numEpochs,  
        printer:VerbosePrinter, 
        namePrefix,
        fold, 
        savePath,
        patience=10,
        min_delta=0.0001,
        early_stopping_epoch=10,
        early_stopping_flag=True,
        schedulerType = "ReduceLROnPlateau",
        random_seed=42):
    frequency = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
    printer(f"Will use the device: {device_name}")
    model.to(device)
    best_loss = float("inf")  # best loss for early stopping
    best_metric = None  # best metric for early stopping
    best_model = None  # best model for early stopping
    is_best_model = False
    train_losses = []  # training loss history
    train_accuracies = []  # training accuracy history
    train_f1_hist = []  # training F1 score history
    torch.manual_seed(random_seed) # set random seed
    evaluator = BinaryF1Score(multidim_average="global").to(device) # evaluator for F1 score
    val_loss_hist = []  # validation loss history
    val_f1_hist = []  # validation F1 score history
    val_acc_hist = []  # validation accuracy history
    start_epoch = 1
    early_stop = False
    # search for pt file start with namePrefix and end with _ckpt.pt
    checkpoint_path = None
    # search for best models of same namePrefix and delete them
    for file in os.listdir(savePath):
        if file.startswith(namePrefix) and file.endswith("_best.pt"):
            os.remove(os.path.join(savePath, file))
    # search for best models of same namePrefix and delete them
    for file in os.listdir(savePath):
        if file.startswith(namePrefix) and file.endswith("_ckpt.pt"):
            checkpoint_path = os.path.join(savePath, file)
            break
    
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.to(device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_metric = checkpoint["best_metric"]
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        printer(f"Loaded the checkpoint from {checkpoint_path}, the epoch is {start_epoch}, the best metric is {best_metric}")
    else:
        printer(f"No checkpoint found at {checkpoint_path}, start training from scratch")
    
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    printer(f"The train dataloader has {len(trainLoader.dataset)} batches")
    printer(f"The validation dataloader has {len(valLoader.dataset)} batches")
    # restore early stopping state when resuming from a checkpoint so that
    # patience / best-tracking does not silently reset
    if checkpoint_path is not None:
        early_stopping.best_metric = best_metric
        if best_loss is not None:
            early_stopping.best_loss = best_loss
    # create the scheduler add additional scheduler if needed
    if schedulerType == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="max",  # monitoring val_accuracy, which should increase
            factor=0.75, 
            patience=10
            )

    elif schedulerType == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=10, 
            gamma=0.1
            )
    elif schedulerType == "None":
        scheduler = None
    else:
        raise ValueError(f"Invalid scheduler type: {schedulerType}")
    
    # create the training loop
    for epoch in range(start_epoch, numEpochs+1):
        model.train()
        train_loss = 0 
        all_train_outputs = []  #
        all_train_labels = []

        for data, cst in trainLoader:
            data, cst = data.to(device), cst.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, cst)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

            all_train_outputs.append(output.detach())
            all_train_labels.append(cst)
        
        avgTrainLoss = train_loss / len(trainLoader.dataset)
        
        train_outputs = torch.cat(all_train_outputs, dim=0)
        train_labels = torch.cat(all_train_labels, dim=0)
        train_f1 = evaluator(train_outputs, train_labels).cpu().numpy()
        train_accuracy = (train_outputs.round() == train_labels).float().mean().item()

        train_losses.append(avgTrainLoss)
        train_accuracies.append(train_accuracy)
        train_f1_hist.append(train_f1)

        # validate the model with the test loader
        model.eval()
        val_loss = 0
        all_val_outputs = []
        all_val_labels = []

        with torch.no_grad():
            for data, cst in valLoader:
                data, cst = data.to(device), cst.to(device)
                output = model(data)
                loss = criterion(output, cst)
                val_loss += loss.item() * data.size(0)

                all_val_outputs.append(output.detach())
                all_val_labels.append(cst)
        
        avgValLoss = val_loss / len(valLoader.dataset)
        val_loss_hist.append(avgValLoss)

        # ReduceLROnPlateau needs the monitored metric (validation loss);
        # StepLR must be stepped without any argument.

        
        val_outputs = torch.cat(all_val_outputs, dim=0)
        val_labels = torch.cat(all_val_labels, dim=0)
        val_f1 = evaluator(val_outputs, val_labels).cpu().numpy()
        val_accuracy = (val_outputs.round() == val_labels).float().mean().item()
        # capture the LR before stepping so a reduction can be detected
        prev_lr = optimizer.param_groups[0]['lr']
        if schedulerType == "ReduceLROnPlateau":
            scheduler.step(val_f1)
        elif schedulerType == "StepLR":
            scheduler.step()
        elif schedulerType == "None":
            pass
        else:
            raise ValueError(f"Invalid scheduler type: {schedulerType}")
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != prev_lr:
            printer(f"Learning rate reduced from {prev_lr:.6f} to {new_lr:.6f}")
        msg = f"{fold} - Epoch {epoch} / {numEpochs}: Train Loss: {avgTrainLoss:.4f}, Train F1: {train_f1:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {avgValLoss:.4f}, Val F1: {val_f1:.4f}, Val Accuracy: {val_accuracy:.4f}"
        printer(msg)
        val_acc_hist.append(val_accuracy)
        val_f1_hist.append(val_f1)
        if epoch % frequency == 0:
            save_checkpoint(model, epoch, optimizer, savePath, namePrefix, best_metric, best_loss)
        # use average train loss and validation F1 score to check the early stopping
        if early_stopping_flag and epoch > early_stopping_epoch:
            early_stop, is_best_model, best_loss, best_metric = early_stopping(val_f1, avgTrainLoss, epoch)
            if not early_stop:
                if is_best_model:
                    printer(f"New best model found at epoch {epoch} with Best F1: {best_metric:.4f}, best loss: {best_loss:.4f}")
                    best_model = deepcopy(model)
                    # save the best model
                    save_best_model(best_model, epoch, best_metric, best_loss, savePath, namePrefix)
                else:
                    printer(f"No best model found at {early_stopping.counter} / {early_stopping.patience} epochs")
            else:
                printer(f"Early stopping triggered at epoch {epoch}")
                break
        
    
    history = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "train_f1_hist": train_f1_hist,
        "val_losses": val_loss_hist,
        "val_accuracies": val_acc_hist,
        "val_f1_hist": val_f1_hist
    }
    # delete any checkpoint files and best model files
    for file in os.listdir(savePath):
        if file.startswith(namePrefix) and file.endswith("_ckpt.pt"):
            os.remove(os.path.join(savePath, file))
        if file.startswith(namePrefix) and file.endswith("_best.pt"):
            os.remove(os.path.join(savePath, file))
    
    if best_model is None:
        # no best model was ever recorded (e.g. numEpochs <= early_stopping_epoch,
        # resumed past numEpochs, or training stopped before reaching numEpochs)
        printer("No best model was recorded during training, returning the last model state")
        best_model = deepcopy(model)
    trained_model = deepcopy(best_model)
    # check if trained model is identical to best model
    if trained_model is not None:
        for param, loaded_param in zip(trained_model.parameters(), best_model.parameters()):
            if param.data.ne(loaded_param.data).sum() > 0:
                raise ValueError(f"The parameter {param.name} is not the same as the best model")
    else:
        raise ValueError(f"The trained model is None")
    printer(f"The trained model is identical to the best model")
    trained_model.to("cpu")
    del best_model, model
    return trained_model, history, device

            
        

        
        
        
        

        
            
        
    
    
    
