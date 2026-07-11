from decoder_trainer import decoder_trainer_validator
from decoder_trainer_tools import NeuralDataset, inference_model, plot_train_history
from dataset_preparer import generate_selection_table, make_random_selection, load_dataset, cnn_reshaper
from neural_drive_decoder import NeuralInterface_1D, NeuralInterface_2D, NeuralInterface_2D_v2, NeuralInterface_3D, NeuralInterface_3D_v2
from utils import VerbosePrinter, setup_logger, list_of_ints, list_of_str, str2bool
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
import argparse
import random

def set_seed(seed: int = 42):
    # Python hash-based randomness (affects set/dict ordering, etc.)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Python's built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU + all GPUs
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # for multi-GPU

    # # Force deterministic algorithms
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

def make_new_selection(args):
    selection_table = generate_selection_table(args.dataset_folder)
    fname = f"RandomSelection_Mix{args.mix_size}_Times{args.times}.json"
    random_selection = make_random_selection(list(selection_table.keys()), args.mix_size, args.times, seed=args.random_seed, saveDir=args.result_folder, fname=fname)
    return random_selection


def main(args):
    startTime = datetime.datetime.now()
    selection_table = generate_selection_table(args.dataset_folder)
    thisFolderName = f"Work-{args.name}_Model-{args.model_type}_Mix-{args.mix_size}_Rep-{args.times}"
    expFolder = os.path.join(args.result_folder, args.name, thisFolderName)
    if not os.path.exists(expFolder):
        os.makedirs(expFolder)
    logger = setup_logger(expFolder)
    printer = VerbosePrinter(logger)
    printer(f"Experiment started at {startTime.strftime('%Y-%m-%d %H:%M:%S')}")
    printer(f"Experiment name: {args.name}")
    printer(f"The number of mix sizes: {args.mix_size}")
    printer(f"Will use the {'raw cst' if args.raw_cst else 'encoded cst'}")
    printer(f"The window size is {args.ws}")
    printer(f"The step size is {args.ss}")
    printer(f"The batch size is {args.batch_size}")
    printer(f"The number of epochs is {args.num_epochs}")
    printer(f"The learning rate is {args.learning_rate}")
    printer(f"The early stopping will be: {'enabled' if args.early_stopping else 'disabled'}")
    printer(f"The early stopping epoch is {args.early_stopping_epoch}")
    printer(f"The patience is {args.patience}")
    printer(f"The minimum delta is {args.min_delta}")
    printer(f"The optimizer is {args.optimizer}")
    printer(f"The learning rate scheduler type is {args.lr_scheduler_type}")
    printer(f"The checkpoint will be: {'enabled' if args.save_checkpoint else 'disabled'}")
    printer(f"The checkpoint frequency is {args.checkpoint_frequency}")
    printer(f"The random seed is {args.random_seed}")
    printer(f"Used the training data from {args.dataset_folder}")
    if args.selection_file is not None:
        printer(f"Will use the selection file: {args.selection_file}")
        with open(args.selection_file, "r") as f:
            selection_record = json.load(f)
            # check if the selection record has the key f"mix_{args.mix_size}"
            if f"mix_{args.mix_size}" in selection_record:
                mix_selections = selection_record[f"mix_{args.mix_size}"]
            else:
                printer(f"The selection record does not have the key mix_{args.mix_size}, will make a new selection")   
                mix_selections_new = make_new_selection(args)
                mix_selections = mix_selections_new[f"mix_{args.mix_size}"]
    else:
        printer(f"No selection file provided, will make a new selection")
        mix_selections_new = make_new_selection(args)
        mix_selections = mix_selections_new[f"mix_{args.mix_size}"]
    
    if len(list(mix_selections.keys())) != args.times:
        printer(f"The number of selections is not equal to the number of times, will reduce the number of times to the number of selections", type="warning")
        args.times = len(list(mix_selections.keys()))
        printer(f"The number of times is now {args.times}")
    
    totalFolds = [i for i in range(1, args.SG+1)]

    # init the seeds
    set_seed(args.random_seed)
    overall_train_result = []

    if args.model_config_file is not None:
        with open(args.model_config_file, "r") as f:
            model_config = json.load(f)
            num_nodes = model_config[args.model_type]
    else:
        if args.model_type == "1D":
            num_nodes = [128, 128, 128, 64, 256]
        elif args.model_type == "2D":
            num_nodes = [64, 64, 64, 64, 256]
        elif args.model_type == "3D":
            num_nodes = [32, 32, 32, 32, 256]
        else:
            raise ValueError(f"Invalid model type: {args.model_type}")

    for trial in range(1, args.times+1):
        printer(f"Training, Validation and Testing for Trial {trial} / {args.times} for Mix Size {args.mix_size}")
        repeatName = f"CNN-{args.model_type}_Mix-{args.mix_size}_Rep-{trial}"
        train_val_dataset = mix_selections[f"Selection{trial}"]["selected_train_dataset"]
        test_dataset = mix_selections[f"Selection{trial}"]["leftover_dataset"]
        test_dataset_files = []
        for dataset in test_dataset:
            test_dataset_files.extend(selection_table[dataset])
        for i, dataset in enumerate(train_val_dataset, start=1):
            print(f"The {i}th train/val dataset of trial {trial} is {dataset}")
        for i, dataset in enumerate(test_dataset_files, start=1):
            print(f"The {i}th test dataset of trial {trial} is {dataset}")
        
        for fold in totalFolds:
            printer(f"Training the model for fold {fold} of trial {trial} for Mix Size {args.mix_size}")
            val_fold = fold
            val_seg = f"SG{val_fold}"
            train_folds = list(filter(lambda x: x != val_fold, totalFolds))
            train_segs = [f"SG{i}" for i in train_folds]
            printer(f"The contractions in the train folds are: {train_folds}")
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
                printer(f"The {i+1}th train dataset file of trial {trial}, fold {fold} is {file}")
            for i, file in enumerate(val_dataset_files):
                printer(f"The {i+1}th val dataset file of trial {trial}, fold {fold} is {file}")
            full_train_dataset = None
            for file in train_dataset_files:
                emg, cst = load_dataset(file, args.raw_cst)
                emg_reshaped = cnn_reshaper(emg, int(args.model_type[0]))
                train_dataset = NeuralDataset(emg_reshaped, cst)
                if full_train_dataset is None:
                    full_train_dataset = train_dataset
                else:
                    full_train_dataset = ConcatDataset([full_train_dataset, train_dataset])
            full_val_dataset = None
            for file in val_dataset_files:
                emg, cst = load_dataset(file, args.raw_cst)
                emg_reshaped = cnn_reshaper(emg, int(args.model_type[0]))
                val_dataset = NeuralDataset(emg_reshaped, cst)
                if full_val_dataset is None:
                    full_val_dataset = val_dataset
                else:
                    full_val_dataset = ConcatDataset([full_val_dataset, val_dataset])
            train_loader = DataLoader(full_train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(full_val_dataset, batch_size=args.batch_size, shuffle=False)
            if args.model_type == "1D":
                model = NeuralInterface_1D(numChannels=64, classes=args.num_output, winLen=args.ws, numNodes=num_nodes)
            elif args.model_type == "2D":
                if args.model_version == "v1":
                    model = NeuralInterface_2D(classes=args.num_output, frameSize=(args.ws, 13, 5), numNodes=num_nodes)
                elif args.model_version == "v2":
                    model = NeuralInterface_2D_v2(classes=args.num_output, frameSize=(args.ws, 13, 5), numNodes=num_nodes)
                else:
                    raise ValueError(f"Invalid model version: {args.model_version}")
                model = NeuralInterface_2D(classes=args.num_output, frameSize=(args.ws, 13, 5), numNodes=num_nodes)
            elif args.model_type == "3D":
                if args.model_version == "v1":
                    model = NeuralInterface_3D(classes=args.num_output, frameSize=(args.ws, 13, 5), numNodes=num_nodes)
                elif args.model_version == "v2":
                    model = NeuralInterface_3D_v2(classes=args.num_output, frameSize=(args.ws, 13, 5), numNodes=num_nodes)
                else:
                    raise ValueError(f"Invalid model version: {args.model_version}")
            else:
                raise ValueError(f"Invalid model type: {args.model_type}")
            printer(f"Using {args.model_type} model with numNodes: {model.numNodes}")
            raw_model = deepcopy(model)
            namePrefix = f"{repeatName}_Fold-{val_fold}"
            displayFoldName = f"{repeatName}_Fold-{val_fold}/{len(totalFolds)}"
            criterion = getattr(torch.nn, args.criterion)()
            optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate, alpha=0.9)
            trained_model, train_result, device = decoder_trainer_validator(
                model = model,
                trainLoader = train_loader,
                valLoader = val_loader,
                criterion = criterion,
                optimizer = optimizer,
                numEpochs = args.num_epochs,
                printer = printer,
                namePrefix = namePrefix,
                fold = displayFoldName,
                savePath = expFolder,
                patience = args.patience,
                min_delta = args.min_delta,
                early_stopping_epoch = args.early_stopping_epoch,
                early_stopping_flag = args.early_stopping,
                schedulerType = args.lr_scheduler_type,
                random_seed = args.random_seed
            )
            result_fname = f"{namePrefix}_train-result.pkl"
            model_fname = f"{namePrefix}_model.pth"
            torch.save(trained_model.state_dict(), os.path.join(expFolder, model_fname))
            joblib.dump(train_result, os.path.join(expFolder, result_fname))
            # plot the train loss history and validation f1 history
            plot_train_history(train_result["train_losses"], expFolder, namePrefix, "train_loss")
            plot_train_history(train_result["val_f1_hist"], expFolder, namePrefix, "val_f1")
            del trained_model, model, train_result
            for file in test_dataset_files:
                emg, cst = load_dataset(file, args.raw_cst)
                emg_reshaped = cnn_reshaper(emg, int(args.model_type[0]))
                test_dataset = NeuralDataset(emg_reshaped, cst)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
                model = deepcopy(raw_model)
                model.load_state_dict(torch.load(os.path.join(expFolder, model_fname)), strict=False)
                model.to(device)
                f1, acc, corr_coeff, rmse, nrmse, pred_cst, bss_cst, smooth_cst_bss, smooth_cst_cnn = inference_model(model, test_loader, device)
                # save the cst s to npz file
                npz_file = os.path.join(expFolder, f"{namePrefix}_{os.path.basename(file).replace('.mat', '.npz')}")
                np.savez(npz_file, raw_cst=bss_cst, pred_cst=pred_cst)
                printer(f"Test result for {os.path.basename(file)}: The F1 score is: {f1:.4f}, the correlation coefficient is: {corr_coeff:.4f}, the RMSE is: {rmse:.4f}, the NRMSE is: {nrmse:.4f}")
                overall_train_result.append({
                    "model_type": args.model_type,
                    "model_version": args.model_version,
                    "mix_size": args.mix_size,
                    "trial": trial,
                    "fold": fold,
                    "R": corr_coeff,
                    "RMSE": rmse,
                    "NRMSE": nrmse,
                    "F1": f1,
                    "Accuracy": acc,
                })
                del file, npz_file, emg, cst, emg_reshaped, test_dataset, test_loader, model
            del raw_model


    result_df = pd.DataFrame(overall_train_result)
    result_df.to_csv(os.path.join(expFolder, f"{namePrefix}_overall_train_result.csv"), index=False)
    result_csv = f"{args.name}_Model-{args.model_type}_Mix-{args.mix_size}_train-result.csv"
    printer(f"Saved the overall train result to {os.path.join(expFolder, result_csv)}")
    endTime = datetime.datetime.now()
    printer(f"Experiment finished at {endTime.strftime('%Y-%m-%d %H:%M:%S')}")
    printer(f"The experiment took {endTime - startTime}")
    # CONVERT THE TIME TO dd:hh:mm:ss format
    time_elapsed_days = (endTime - startTime).days
    time_elapsed_hours = (endTime - startTime).seconds // 3600
    time_elapsed_minutes = (endTime - startTime).seconds // 60 % 60
    time_elapsed_seconds = (endTime - startTime).seconds % 60
    printer(f"The experiment took {time_elapsed_days} days, {time_elapsed_hours} hours, {time_elapsed_minutes} minutes, {time_elapsed_seconds} seconds")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default="Exp7", help="The name of the experiment")
    parser.add_argument("--random_seed", type=int, default=42, help="The random seed for the experiment")

    # define necessary directories
    parser.add_argument("--model_config_file", type=str, default=r"G:\NMI_JNE_Validation\Training_LOOCV\num_nodes.json", help="The path to the model config file")
    parser.add_argument("--selection_file", type=str, default=r"G:\NMI_JNE_Validation\Old_Protocol\RandomSelection_Mix1To19_Times10.json", help="The path to the selection file")
    parser.add_argument("--dataset_folder", type=str, default=r"G:\NMI_Journal_Dataset\organized\Train", help="The directory of the dataset")
    parser.add_argument("--result_folder", type=str, default=r"G:\NMI_JNE_Validation\Old_Protocol", help="The directory to save the training results")

    # Train Parameters
    parser.add_argument("--mix_size", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], default=1, help="The size of the mix for training CNN")
    parser.add_argument("--times", type=int, default=10, help="The number of times to make the selection")
    parser.add_argument("--raw_cst", type=str2bool, default=True, help="Whether to use the raw CST in the .mat file of the dataset, false will use the encoded CST")
    parser.add_argument("--ws", type=int, default=40, help="The window size for training CNN")
    parser.add_argument("--ss", type=int, default=20, help="The step size for training CNN")
    parser.add_argument("--SG", type=int, default=3, help="The number of contractions of each dataset, will be used for cross-validation")

    # Model Parameters
    parser.add_argument("--model_type", type=str, choices=["1D", "2D", "3D"], default="1D", help="The type of the model to use")
    parser.add_argument("--model_version", type=str, choices=["v1", "v2", "v3"], default="v1", help="The version of the model to use")
    parser.add_argument("--num_output", type=int, default=4, help="The number of output classes for the model")

    # Training Parameters
    parser.add_argument("--batch_size", type=int, default=128, help="The batch size for training CNN")
    parser.add_argument("--num_epochs", type=int, default=200, help="The number of epochs for training CNN")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="The learning rate for training CNN")
    parser.add_argument("--early_stopping", type=str2bool, default=True, help="Whether to use early stopping")
    parser.add_argument("--early_stopping_epoch", type=int, default=50, help="The number of epochs to wait before early stopping")
    parser.add_argument("--patience", type=int, default=50, help="The patience for early stopping")
    parser.add_argument("--min_delta", type=float, default=0.001, help="The minimum delta for early stopping")
    parser.add_argument("--criterion", type=str, choices=["BCELoss", "MSELoss"], default="BCELoss", help="The criterion for training CNN")
    parser.add_argument("--optimizer", type=str, choices=["Adam", "AdamW", "SGD", "RMSprop"], default="RMSprop", help="The optimizer for training CNN")
    parser.add_argument("--lr_scheduler_type", type=str, choices=["StepLR", "ExponentialLR", "CosineAnnealingLR", "MultiStepLR", "ReduceLROnPlateau"], default="ReduceLROnPlateau", help="The type of the learning rate scheduler")
    parser.add_argument("--lr_scheduler_step_size", type=int, default=30, help="The step size for the StepLR scheduler")
    parser.add_argument("--save_checkpoint", type=str2bool, default=True, help="Whether to save the checkpoint")
    parser.add_argument("--checkpoint_frequency", type=int, default=10, help="The frequency of saving the checkpoint")
    args = parser.parse_args()

    main(args)
