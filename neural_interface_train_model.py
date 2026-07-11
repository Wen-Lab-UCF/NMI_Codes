from decoder_trainer import decoder_trainer_validator
from decoder_trainer_tools import NeuralDataset, inference_model, plot_train_history
from dataset_preparer import generate_selection_table, make_random_selection, load_dataset, cnn_reshaper
from neural_drive_decoder import NeuralInterface_1D, NeuralInterface_2D, NeuralInterface_2D_v2, NeuralInterface_3D, NeuralInterface_3D_v2
from utils import VerbosePrinter, setup_logger
import datetime
import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset, DataLoader
from copy import deepcopy
import joblib
import pandas as pd
import json
from glob import glob
# Set up the experiment parameters, make random selection of the dataset
dataset_dir = r"G:\NMI_Journal_Dataset\organized\Train"
result_dir_root = r"G:\NMI_JNE_Validation\Old_Protocol"
os.makedirs(result_dir_root, exist_ok=True)
selection_file = None
modelTypes = ["1D", "2D", "3D"]
modelVersion = "v1"
mix_size = 11
repetitions = 5
SG = 3
WS = 40
ST = 20
batch_size = 64
learning_rate = 0.01
num_epochs = 100
patience = 20
min_delta = 0.001
early_stopping_epoch = 50
early_stopping_flag = True
scheduler_type = "None"
random_seed = 42
overall_train_result = []
result_dir_mix = os.path.join(result_dir_root, f"Mix{mix_size}_Rep{repetitions}_Model{modelVersion}")

taskTimeStamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
for modelType in modelTypes:
    folderName = f"CNN{modelType}_Model{modelVersion}_Mix{mix_size}_Rep{repetitions}"
    result_dir = os.path.join(result_dir_mix, folderName)
    os.makedirs(result_dir, exist_ok=True)
    logger = setup_logger(result_dir)
    printer = VerbosePrinter(logger)

    ext = ".mat"
    selection_table = generate_selection_table(dataset_dir)

    # search for json filr start with RandomChoice_mix{mix_size}_rep{repetitions}
    json_files = glob(os.path.join(result_dir_root, "**/*.json"), recursive=True)
    for json_file in json_files:
        if f"RandomChoice_mix{mix_size}_rep{repetitions}" in json_file:
            selection_file = json_file
            break
    if selection_file is None:
        random_choice_fname = f"RandomChoice_mix{mix_size}_rep{repetitions}.json"
        random_choices = make_random_selection(list(selection_table.keys()), mix_size, repetitions, saveDir=result_dir, fname=random_choice_fname)
    else:
        with open(selection_file, "r") as f:
            random_choices = json.load(f)
    for key, value in random_choices.items():
        printer(f"Selection {key}")
        for i, choice in enumerate(value["selected_train_dataset"]):
            printer(f"Selection {i+1}: {choice}")

    selections = list(random_choices.keys())

    totalFolds = [i for i in range(1, SG+1)]

    
    printer(f"Start the experiment for {modelType} CNN, Mix size {mix_size}, with {repetitions} repetitions, WS {WS}, ST {ST}")
    for repetition in range(len(selections)):
        repeat_name = f"CNN-{modelType}_Mix-{mix_size}_Rep-{repetition+1}"
        printer(f"Repetition {repetition+1} of Mix {mix_size}")
        train_val_dataset = random_choices[selections[repetition]]["selected_train_dataset"]
        printer(f"The training and validation datasets are: {train_val_dataset}")
        test_dataset = random_choices[selections[repetition]]["leftover_dataset"]
        printer(f"The test datasets are: {test_dataset}")
        test_dataset_files = []
        for dataset in test_dataset:
            test_dataset_files.extend(selection_table[dataset])

        for i, file in enumerate(test_dataset_files):
            printer(f"The test dataset file {i+1} is: {file}")
        for fold in totalFolds:
            printer(f"Training the model for fold {fold}")
            val_fold = fold
            val_seg = f"SG{val_fold}"
            train_folds = list(filter(lambda x: x != val_fold, totalFolds))
            train_segs = [f"SG{i}" for i in train_folds]
            printer(f"The train folds are: {train_folds}")
            printer(f"The val fold is: {val_fold}")
            train_dataset_files = []
            val_dataset_files = []
            for dataset in train_val_dataset:
                files = selection_table[dataset]
                for file in files:
                    if val_seg in file:
                        val_dataset_files.append(file)
                    else:
                        train_dataset_files.append(file)
            for i, file in enumerate(train_dataset_files):
                printer(f"The train dataset file {i+1} is: {file}")
            for i, file in enumerate(val_dataset_files):
                printer(f"The val dataset file {i+1} is: {file}")
            
            full_train_dataset = None
            for file in train_dataset_files:
                emg, cst = load_dataset(file)
                emg_reshaped = cnn_reshaper(emg, int(modelType[0]))
                # build train dataset
                train_dataset = NeuralDataset(emg_reshaped, cst)
                if full_train_dataset is None:
                    full_train_dataset = train_dataset
                else:
                    full_train_dataset = ConcatDataset([full_train_dataset, train_dataset])
            
            full_val_dataset = None
            for file in val_dataset_files:
                emg, cst = load_dataset(file)
                emg_reshaped = cnn_reshaper(emg, int(modelType[0]))
                val_dataset = NeuralDataset(emg_reshaped, cst)
                if full_val_dataset is None:
                    full_val_dataset = val_dataset
                else:
                    full_val_dataset = ConcatDataset([full_val_dataset, val_dataset])
            train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(full_val_dataset, batch_size=batch_size, shuffle=True)

            if modelType == "1D":
                model = NeuralInterface_1D(numChannels=64, classes=4, winLen=40, numNodes=[128, 128, 128, 64, 256])  # v1: [128, 128, 128, 64, 256], v2: [64, 64, 64, 32, 128]
                printer(f"Using 1D model with numNodes: {model.numNodes}")
            elif modelType == "2D":
                if modelVersion == "v1":
                    model = NeuralInterface_2D(classes=4, frameSize=(40, 13, 5), numNodes=[64, 64, 64, 64, 256])  # v1: [64, 64, 64, 64, 256], v2: [64, 64, 64, 32, 128]
                elif modelVersion == "v2":
                    model = NeuralInterface_2D_v2(classes=4, frameSize=(40, 13, 5), numNodes=[64, 64, 64, 32, 128])
                printer(f"Using 2D model with numNodes: {model.numNodes}")
            elif modelType == "3D":
                if modelVersion == "v1":
                    model = NeuralInterface_3D(classes=4, frameSize=(1, 40, 13, 5), numNodes=[32, 32, 32, 32, 256])  # v1: [32, 32, 32, 32, 256], v2: [16, 16, 16, 16, 128]
                elif modelVersion == "v2":
                    model = NeuralInterface_3D_v2(classes=4, frameSize=(1, 40, 13, 5), numNodes=[64, 64, 64, 32, 128])
                printer(f"Using 3D model with numNodes: {model.numNodes}")
            else:
                raise ValueError(f"Invalid model type: {model}")

            raw_model = deepcopy(model)
            
            namePrefix = f"{repeat_name}_Fold-{val_fold}"
            
            criterion = torch.nn.BCELoss()
            optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)
            
            trained_model, train_result, device = decoder_trainer_validator(
                model=model, 
                trainLoader=train_loader, 
                valLoader=val_loader, 
                criterion=criterion,
                fold=fold,
                optimizer=optimizer, 
                numEpochs=num_epochs, 
                printer=printer, 
                namePrefix=namePrefix, 
                savePath=result_dir, 
                patience=patience, 
                min_delta=min_delta, 
                early_stopping_epoch=early_stopping_epoch, 
                early_stopping_flag=early_stopping_flag, 
                schedulerType=scheduler_type, 
                random_seed=random_seed)
            result_fname = f"{namePrefix}_train_result.pkl"
            model_fname = f"{namePrefix}_model.pth"
            torch.save(trained_model.state_dict(), os.path.join(result_dir, model_fname))
            joblib.dump(train_result, os.path.join(result_dir, result_fname))
            # plot train loss history, validation f1
            plot_train_history(train_result["train_losses"], result_dir, namePrefix, "train_loss")
            plot_train_history(train_result["val_f1_hist"], result_dir, namePrefix, "val_f1")
            del trained_model, model, train_result
            for file in test_dataset_files:
                emg, cst = load_dataset(file)
                emg_reshaped = cnn_reshaper(emg, int(modelType[0]))
                test_dataset = NeuralDataset(emg_reshaped, cst)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                model = deepcopy(raw_model)
                model.load_state_dict(torch.load(os.path.join(result_dir, model_fname)), strict=False)
                model.to(device)
                outputs, labels, f1, accuracy, corr_coeff, rmse, nrmse, cst_bss_scaled, cst_cnn_scaled, smooth_cst_bss, smooth_cst_cnn = inference_model(model, test_loader, device, 25)
                # save the cst s to npz file
                npz_file = os.path.join(result_dir, f"{namePrefix}_{os.path.basename(file).replace('.mat', '.npz')}")
                np.savez(npz_file, raw_cst_scaled=cst_bss_scaled, pred_cst_scaled=cst_cnn_scaled, raw_cst=labels, pred_cst=outputs)
                printer(f"Test result for {os.path.basename(file)}: The F1 score is: {f1:.4f}, the correlation coefficient is: {corr_coeff:.4f}, the RMSE is: {rmse:.4f}")
                overall_train_result.append({
                    "model_type": modelType,
                    "model_version": modelVersion,
                    "mix_size": mix_size,
                    "WS": WS,
                    "SS": ST,
                    "repetition": repetition+1,
                    "fold": fold,
                    "test_file": os.path.basename(file),
                    "test_r": corr_coeff,
                    "test_rmse": rmse,
                    "test_f1": f1,
                    "test_accuracy": accuracy,
                })
    

result_df = pd.DataFrame(overall_train_result)
result_df.to_csv(os.path.join(result_dir_root, "overall_train_result.csv"), index=False)




