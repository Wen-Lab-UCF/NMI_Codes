from ast import Break
from decoder_trainer import decoder_trainer_validator
from decoder_trainer_tools import NeuralDataset, inference_model, inference_model_simple_version
from dataset_preparer import spikes2cst, encode_cst, cnn_reshaper
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
import gc
import argparse
import shutil

def main(args):
    trainResultDir = args.train_result_dir
    inferenceDataDir = args.inference_data_dir
    inferenceResultDirRoot = args.inference_result_dir_root
    if args.inference_name is not None:
        inferenceResultDir = os.path.join(inferenceResultDirRoot, os.path.basename(os.path.normpath(trainResultDir)), args.inference_name)
    else:
        inferenceResultDir = os.path.join(inferenceResultDirRoot, os.path.basename(os.path.normpath(trainResultDir)))
    if not os.path.exists(inferenceResultDir):
        os.makedirs(inferenceResultDir)
    logger = setup_logger("Inference_loocv", log_dir=inferenceResultDir)
    printer = VerbosePrinter(logger)

    modelFiles = glob(os.path.join(trainResultDir, "**/*.pth"), recursive=True)
    printer(f"Found {len(modelFiles)} model files")
    inference_data_list = glob(os.path.join(inferenceDataDir, "**/*.pkl"), recursive=True)
    SG = 3
    # inferenceResultDir = os.path.join(inferenceResultDirRoot, os.path.basename(os.path.normpath(trainResultDir)))
    # if not os.path.exists(inferenceResultDir):
    #     os.makedirs(inferenceResultDir)

    nodeFile = os.path.join(trainResultDir, "num_nodes.json")
    shutil.copy(nodeFile, os.path.join(inferenceResultDir, "num_nodes.json"))
    with open(nodeFile, "r") as f:
        numNodesDict = json.load(f)

    for modelFile in modelFiles:
        modelFname = os.path.basename(modelFile)
        modelDir = os.path.dirname(modelFile)
        printer(f"Processing {modelFname}")
        modelNameParts = modelFname.split("_")
        modelType = int(modelNameParts[0].split("-")[1][0])
        REPETITION = int(modelNameParts[2].split("-")[1])
        FOLD = int(modelNameParts[3].split("-")[1])
        printer(f"Model type: {modelType}, Repetition: {REPETITION}, Fold: {FOLD}")
        if modelType == 1:
            numNodes = numNodesDict["1D"]
            model = NeuralInterface_1D(numChannels=64, classes=4, winLen=40, numNodes=numNodes)
            printer(f"Using 1D model with numNodes: {model.numNodes}")
        elif modelType == 2:
            numNodes = numNodesDict["2D"]
            model = NeuralInterface_2D(classes=4, frameSize=(40, 13, 5), numNodes=numNodes)
            # model = NeuralInterface_2D(classes=4, frameSize=(40, 13, 5), numNodes=[64, 64, 64, 32, 128])
            printer(f"Using 2D model with numNodes: {model.numNodes}")
        elif modelType == 3:
            numNodes = numNodesDict["3D"]
            model = NeuralInterface_3D(classes=4, frameSize=(1, 40, 13, 5), numNodes=numNodes)
            # model = NeuralInterface_3D(classes=4, frameSize=(1, 40, 13, 5), numNodes=[32, 32, 32, 16, 64])
            printer(f"Using 3D model with numNodes: {model.numNodes}")
        else:
            raise ValueError(f"Invalid model type: {modelType}")

        model.load_state_dict(torch.load(modelFile))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        printer(f"Will use the device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}for inference")
        model.to(device)
        for pkl_file in inference_data_list:
            fileName = os.path.basename(pkl_file)
            printer(f"Processing {fileName}")
            fileNameParts = fileName.split("_")
            datasetType = fileNameParts[0]
            session = fileNameParts[1][1]
            subject = fileNameParts[2][2:]
            intensity = int(fileNameParts[3])
            muscle = fileNameParts[4]
            printer(f"Testing dataset: {datasetType}, Session: {session}, Subject: {subject}, Intensity: {intensity}, Muscle: {muscle}")
            data = joblib.load(pkl_file)
            for seg in range(1, SG+1):
                segment = seg
                segmentData = data[segment];
                emg = segmentData["EMGs"]
                spikes = segmentData["Spikes"]
                cst_bss = spikes2cst(spikes)
                cst_bss_raw = cst_bss.copy()
                cst_trimmed = encode_cst(cst_bss, num_outputs=4)
                emg_reshaped = cnn_reshaper(emg, modelType)
                printer(f"The shape of the loaded EMG is {emg.shape}, loaded spikes is {spikes.shape}, reshaped EMG is {emg_reshaped.shape}, encoded CST is {cst_trimmed.shape}")
                test_dataset = NeuralDataset(emg_reshaped, cst_trimmed)
                test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
                pred_cst, bss_cst, _, _ = inference_model_simple_version(model, test_loader, device)
                cst_name = f"CNN-{modelType}_N-20_R-{REPETITION}_F-{FOLD}_T-{datasetType}_D-{session}_S-{subject}_I-{intensity}_M-{muscle}_SG-{seg}.csv"
                local_result_df = pd.DataFrame({
                    "CST_BSS": np.asarray(bss_cst).ravel().astype(np.float64),
                    "CST_BSS_RAW": np.asarray(cst_bss_raw).ravel().astype(np.float64),
                    "CST_CNN": np.asarray(pred_cst).ravel().astype(np.float64),
                })
                local_result_df.to_csv(os.path.join(inferenceResultDir, cst_name), index=False)
                printer(f"Saved the result to {os.path.join(inferenceResultDir, cst_name)}")
        del model
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
<<<<<<< HEAD
    parser.add_argument("--inference_name", type=str, default=None, help="The name of the inference")
    parser.add_argument("--train_result_dir", type=str, default=r"G:\Dropbox\1_NMI-Projects\1_Resubmission-Results\Train_NewPadding_SameNodes\LOOCV_Correct_Padding_Same_Filters", help="The directory to the training result")
    parser.add_argument("--inference_data_dir", type=str, default=r"G:\NMI_Journal_Dataset\prepared\Evaluation", help="The directory to the inference data")
    parser.add_argument("--inference_result_dir_root", type=str, default=r"G:\Dropbox\1_NMI-Projects\1_Resubmission-Results\Train_NewPadding_SameNodes\Inference", help="The directory to save the inference result")
=======
    parser.add_argument("--inference_name", type=str, default="Inference_LOOCV", help="The name of the inference")
    parser.add_argument("--train_result_dir", type=str, default=r"C:\Users\wenlab_pc_user\Downloads\LOOCV_1D_Validate", help="The directory to the training result")
    parser.add_argument("--inference_data_dir", type=str, default=r"E:\NMI_Dataset\prepared\Evaluation", help="The directory to the inference data")
    parser.add_argument("--inference_result_dir_root", type=str, default=r"C:\Users\wenlab_pc_user\Downloads\Inference_Results", help="The directory to save the inference result")
>>>>>>> 06eb573703296900f2bce24600f4d9a78338c316
    args = parser.parse_args()
    main(args)