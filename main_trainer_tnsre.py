from decoder_trainer import decoder_trainer_validator
from decoder_trainer_tools import NeuralDataset, inference_model, plot_train_history
from dataset_preparer import generate_selection_table, make_random_selection, load_dataset, cnn_reshaper
from neural_drive_decoder import NeuralInterface_1D, NeuralInterface_2D, NeuralInterface_3D
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
from torchinfo import summary
import io
from contextlib import redirect_stdout

def print_model_summary(model, dummy_input, printer, savepath):
    col_names = ("input_size", "output_size", "num_params", "kernel_size", "mult_adds")
    with io.StringIO() as buf, redirect_stdout(buf):
        printer(summary(model, input_data=dummy_input, col_names=col_names))
        summary_text = buf.getvalue()
    with open(savepath, "w", encoding="utf-8") as f:
        f.write(summary_text)
LOOCV_FOLDS = {"Fold-1":{"Train":["d1_VL", "d1_VM", "d2_VL"], "Test":["d2_VM"]},
               "Fold-2":{"Train":["d1_VL", "d1_VM", "d2_VM"], "Test":["d2_VL"]},
               "Fold-3":{"Train":["d1_VL", "d2_VM", "d2_VL"], "Test":["d1_VM"]},
               "Fold-4":{"Train":["d1_VM", "d2_VM", "d2_VL"], "Test":["d1_VL"]}
               }

dataset_dir = r"C:\Users\wenlab_pc_user\Downloads\Data"
result_dir_root = r"C:\Users\wenlab_pc_user\Downloads\valid_1dcnn"
os.makedirs(result_dir_root, exist_ok=True)
modelTypes = ["1D"]
modelVersion = "v1"  #
SG = 3
WS = 40
ST = 20
subjects = 5
batch_size = 128
learning_rate = {"1D": 0.01, "2D": 0.001, "3D": 0.001}
num_epochs = {"1D": 100, "2D": 150, "3D": 200}
patience = 20
min_delta = 0.001
early_stopping_epoch = 50
early_stopping_flag = True
scheduler_type = "ReduceLROnPlateau"
random_seed = 42
overall_train_result = []

taskTimeStamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
logger = setup_logger(f"LOOCV_Model{modelVersion}", result_dir_root)
printer = VerbosePrinter(logger)

FOLDS = list(LOOCV_FOLDS.keys())
printer(f"The folds are: {FOLDS}")

num_nodes = {"1D": [128, 128, 128, 64, 256], "2D": [128, 128, 128, 64, 256], "3D": [128, 128, 128, 64, 256]}
# save it as a json file
with open(os.path.join(result_dir_root, "num_nodes.json"), "w") as f:
    json.dump(num_nodes, f)

for modelType in modelTypes:
    result_dir = os.path.join(result_dir_root, f"LOOCV_{modelVersion}Model_CNN{modelType}")
    os.makedirs(result_dir, exist_ok=True)
    printer(f"Training the {modelType} CNN model using LOOCV Mode")
    if modelType == "1D":
        standard_model = NeuralInterface_1D(numChannels=64, classes=4, winLen=40, numNodes=num_nodes[modelType])  # v1: [128, 128, 128, 64, 256], v2: [64, 64, 64, 32, 128]
        printer(f"Using 1D model with numNodes: {standard_model.numNodes}")
        print_model_summary(standard_model, torch.zeros([64, 64, 40]), printer, os.path.join(result_dir, f"model_1dcnn_summary.txt"))
    elif modelType == "2D":
        standard_model = NeuralInterface_2D(classes=4, frameSize=(40, 13, 5), numNodes=num_nodes[modelType])  # v1: [64, 64, 64, 32, 256], v2: [64, 64, 64, 32, 128]
        print_model_summary(standard_model, torch.zeros([64, 40, 13, 5]), printer, os.path.join(result_dir, f"model_2dcnn_summary.txt"))
    elif modelType == "3D":
        standard_model = NeuralInterface_3D(classes=4, frameSize=(1, 40, 13, 5), numNodes=num_nodes[modelType])  # v1: [32, 32, 32, 32, 256], v2: [16, 16, 16, 16, 128]
        print_model_summary(standard_model, torch.zeros([64, 1, 40, 13, 5]), printer, os.path.join(result_dir, f"model_3dcnn_summary.txt"))
    for nF, fold in enumerate(FOLDS, start=1):
        printer(f"Training the {modelType} CNN model for {fold}")
        repeat_name = f"CNN-{modelType}_LOOCV_Fold-{nF}"
        train_folds = LOOCV_FOLDS[fold]["Train"]
        test_folds = LOOCV_FOLDS[fold]["Test"]
        train_val_datasets = {}
        for segment in range(1, SG+1):
            train_val_datasets[f"Run-{segment}"] = {"Train": [], "Val": []}
        test_datasets = []
        for subject in range(1, subjects+1):
            for train_fold in train_folds:
                session, muscle = train_fold.split("_")
                for segment in range(1, SG+1):
                    val_seg = segment
                    train_segs = [i for i in range(1, SG+1) if i != segment]
                    for train_seg in train_segs:
                        fname = f"Train_{session}_uc{subject}_25_{muscle}-SG{train_seg}-WS{WS}-ST{ST}.mat"
                        printer(f"Construct the training dataset {fname} for fold {fold}")
                        if os.path.exists(os.path.join(dataset_dir, fname)):
                            train_val_datasets[f"Run-{segment}"]["Train"].append(os.path.join(dataset_dir, fname))
                        else:
                            raise FileNotFoundError(f"The file {fname} does not exist")
                    fname_val = f"Train_{session}_uc{subject}_25_{muscle}-SG{val_seg}-WS{WS}-ST{ST}.mat"
                    printer(f"Construct the validation dataset {fname_val} for fold {fold}")
                    if os.path.exists(os.path.join(dataset_dir, fname_val)):
                        train_val_datasets[f"Run-{segment}"]["Val"].append(os.path.join(dataset_dir, fname_val))
                    else:
                        raise FileNotFoundError(f"The file {fname_val} does not exist")

            for test_fold in test_folds:
                for segment in range(1, SG+1):
                    session, muscle = test_fold.split("_")
                    fname = f"Train_{session}_uc{subject}_25_{muscle}-SG{segment}-WS{WS}-ST{ST}.mat"
                    printer(f"Construct the test dataset {fname} for fold {fold}")
                    if os.path.exists(os.path.join(dataset_dir, fname)):
                        test_datasets.append(os.path.join(dataset_dir, fname))
                    else:
                        raise FileNotFoundError(f"The file {fname} does not exist")
        
        run_list = list(train_val_datasets.keys())

        for nR, run in enumerate(run_list, start=1): 
            model = deepcopy(standard_model)
            printer(f"Copy the standard model to the model for training")
            # check the model is untrained
            if not model.trained:
                printer(f"The model is untrained, will train the model")
            else:
                raise ValueError(f"The model is trained, check the code to make sure for each fold and eacu subfold, the model is untrained")
            train_dataset = None
            val_dataset = None
            train_data_files = train_val_datasets[run]["Train"]
            printer(f"The train data includes {len(train_data_files)} files which are")
            for file in train_data_files:
                printer(f"    {file}")
            val_data_files = train_val_datasets[run]["Val"]
            printer(f"The val data includes {len(val_data_files)} files which are")
            for file in val_data_files:
                printer(f"    {file}")
            
            train_dataset = None
            val_dataset = None
            for train_data_file in train_data_files:
                emg, cst, raw_cst = load_dataset(train_data_file)
                emg_reshaped = cnn_reshaper(emg, int(modelType[0]))
                train_neural_dataset = NeuralDataset(emg_reshaped, cst)
                if train_dataset is None:
                    train_dataset = train_neural_dataset
                else:
                    train_dataset = ConcatDataset([train_dataset, train_neural_dataset])
            for val_data_file in val_data_files:
                emg, cst, _ = load_dataset(val_data_file)
                emg_reshaped = cnn_reshaper(emg, int(modelType[0]))
                val_neural_dataset = NeuralDataset(emg_reshaped, cst)
                if val_dataset is None:
                    val_dataset = val_neural_dataset
                else:
                    val_dataset = ConcatDataset([val_dataset, val_neural_dataset])
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            raw_model = deepcopy(model)
            criterion = torch.nn.BCELoss()
            if modelType == "1D":
                optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate[modelType], alpha=0.9)
            elif modelType == "2D":
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate[modelType])
            elif modelType == "3D":
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate[modelType])
            else:
                raise ValueError(f"Invalid model type: {modelType}")
            namePrefix = f"{repeat_name}_Run-{nR}"
            trained_model, train_result, device = decoder_trainer_validator(
                model=model, 
                trainLoader=train_loader, 
                valLoader=val_loader, 
                criterion=criterion,
                fold=f"CNN_{modelType}_Fold_{nF}-Run_{nR}",
                optimizer=optimizer, 
                numEpochs=num_epochs[modelType], 
                printer=printer, 
                namePrefix=namePrefix, 
                savePath=result_dir, 
                patience=patience, 
                min_delta=min_delta, 
                early_stopping_epoch=early_stopping_epoch, 
                early_stopping_flag=early_stopping_flag, 
                schedulerType=scheduler_type, 
                random_seed=random_seed)
            trained_model.trained = True
            result_fname = f"{namePrefix}_train_result.pkl"
            model_fname = f"{namePrefix}_model.pth"
            # add number of nodes to the trained_model's state_dict
            torch.save(trained_model.state_dict(), os.path.join(result_dir, model_fname))
            joblib.dump(train_result, os.path.join(result_dir, result_fname))
            # plot train loss history, validation f1
            plot_train_history(train_result["train_losses"], result_dir, namePrefix, "train_loss")
            plot_train_history(train_result["val_f1_hist"], result_dir, namePrefix, "val_f1")
            del trained_model, model, train_result
            for file in test_datasets:
                emg, cst, raw_cst = load_dataset(file)
                # squeeze the raw_cst to 1D
                raw_cst = raw_cst.squeeze()
                emg_reshaped = cnn_reshaper(emg, int(modelType[0]))
                printer(f"The shape of the test emg is {emg_reshaped.shape}")
                test_neural_dataset = NeuralDataset(emg_reshaped, cst)
                test_loader = DataLoader(test_neural_dataset, batch_size=batch_size, shuffle=False)
                model = deepcopy(raw_model)
                model.load_state_dict(torch.load(os.path.join(result_dir, model_fname)), strict=False)
                model.to(device)
                f1, accuracy, corr_coeff, rmse, nrmse, pred_cst, bss_cst, smooth_cst_bss, smooth_cst_cnn = inference_model(model, test_loader, device)
                # save the cst s to csv file
                csv_file = os.path.join(result_dir, f"{namePrefix}_{os.path.basename(file).replace('.mat', '.csv')}")
                pd.DataFrame({
                    "RAW_CST_LABEL": bss_cst,
                    "RAW_CST_UNTRIMMED": raw_cst,
                    "CNN_OUTPUT_CST": pred_cst,
                    "smooth_cst_bss": smooth_cst_bss,
                    "smooth_cst_cnn": smooth_cst_cnn,
                }).to_csv(csv_file, index=False)
                printer(f"Test result for {os.path.basename(file)}: The F1 score is: {f1:.4f}, the correlation coefficient is: {corr_coeff:.4f}, the RMSE is: {rmse:.4f}, the NRMSE is: {nrmse:.4f}")
                overall_train_result.append({
                    "model_type": modelType,
                    "model_version": modelVersion,
                    "mix_size": 20,
                    "WS": WS,
                    "SS": ST,
                    "LOOCV_FOLD": nF,
                    "3FOLD_FOLD": nR,
                    "test_file": os.path.basename(file),
                    "test_r": corr_coeff,
                    "test_rmse": rmse,
                    "test_nrmse": nrmse,
                    "test_f1": f1,
                    "test_accuracy": accuracy,
                })
                del emg, cst, emg_reshaped, test_neural_dataset, test_loader, model
            del raw_model


result_df = pd.DataFrame(overall_train_result)
result_df.to_csv(os.path.join(result_dir_root, "overall_train_result.csv"), index=False)

