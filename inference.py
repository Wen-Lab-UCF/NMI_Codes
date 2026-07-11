import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from glob import glob
from neural_drive_decoder import NeuralInterface_1D, NeuralInterface_2D, NeuralInterface_3D
from decoder_trainer_tools import inference_model, NeuralDataset
from utils import VerbosePrinter, setup_logger
import torch
from dataset_preparer import spikes2cst, encode_cst, cnn_reshaper
from torch.utils.data import DataLoader
# import minmax scaler

result_dir = r"G:\NMI_JNE_Validation\Mix11_Rep5_Modelv1"
save_dir = r"G:\NMI_JNE_Validation_Inference"
data_dir = r"G:\NMI_Journal_Dataset\prepared\Evaluation"
SG = 3

logger = setup_logger("Inference", log_dir=save_dir)
printer = VerbosePrinter(logger)

printer("Start inference")

# search for pth files recursively in the result_dir
pth_files = glob(os.path.join(result_dir, "**/*.pth"), recursive=True)
printer(f"Found {len(pth_files)} checkpoint files")
inference_result = []

for pth_file in pth_files:
    fileName = os.path.basename(pth_file)
    printer(f"Processing {fileName}")
    fileNameParts = fileName.split("_")
    modelType = fileNameParts[0]
    modelType = int(modelType.split("-")[1][0])
    mix = fileNameParts[1].split("-")[1]
    repetition = fileNameParts[2].split("-")[1]
    fold = fileNameParts[3].split("-")[1]
    printer(f"Run inference for the {modelType}DCNN model with mix {mix} and repetition {repetition} and fold {fold}")
    if modelType == 1:
        model = NeuralInterface_1D(numChannels=64, classes=4, winLen=40, numNodes=[128, 128, 128, 64, 256])
        printer(f"Using 1D model with numNodes: {model.numNodes}")
    elif modelType == 2:
        model = NeuralInterface_2D(classes=4, frameSize=(40, 13, 5), numNodes=[64, 64, 64, 64, 256])
        printer(f"Using 2D model with numNodes: {model.numNodes}")
    elif modelType == 3:
        model = NeuralInterface_3D(classes=4, frameSize=(1, 40, 13, 5), numNodes=[32, 32, 32, 32, 256])
        printer(f"Using 3D model with numNodes: {model.numNodes}")
    else:
        raise ValueError(f"Invalid model type: {modelType}")
    model.load_state_dict(torch.load(pth_file))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    printer(f"Will use the device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}for inference")
    model.to(device)
    # search for pkl files recursively in the data_dir
    pkl_files = glob(os.path.join(data_dir, "**/*.pkl"), recursive=True)
    printer(f"Found {len(pkl_files)} pkl files")
    save_folder_name = f"CNN{modelType}D_Mix{mix}_Rep{repetition}_Fold{fold}"
    save_folder = os.path.join(save_dir, save_folder_name)
    os.makedirs(save_folder, exist_ok=True)
    for pkl_file in pkl_files:
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
            cst = spikes2cst(spikes)
            cst = encode_cst(cst, num_outputs=4)
            emg_reshaped = cnn_reshaper(emg, modelType)
            printer(f"The shape of the loaded EMG is {emg.shape}, loaded spikes is {spikes.shape}, reshaped EMG is {emg_reshaped.shape}, encoded CST is {cst.shape}")
            test_dataset = NeuralDataset(emg_reshaped, cst)
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            pred_cst, bss_cst, f1, accuracy, corr_coeff, rmse, cst_bss_scaled, cst_cnn_scaled, smooth_cst_bss, smooth_cst_cnn = inference_model(model, test_loader, device, intensity)
            cst_name = f"CNN{modelType}D_Mix{mix}_Rep{repetition}_Fold{fold}_{datasetType}_d{session}_uc{subject}_{intensity}_{muscle}_SG{seg}.npz"
            cst_path = os.path.join(save_folder, cst_name)
            np.savez(cst_path, raw_cst_scaled=cst_bss_scaled, pred_cst_scaled=cst_cnn_scaled, raw_cst=bss_cst, pred_cst=pred_cst)
            smooth_cst_bss_minmax = (smooth_cst_bss - np.min(smooth_cst_bss)) / (np.max(smooth_cst_bss) - np.min(smooth_cst_bss))
            smooth_cst_cnn_minmax = (smooth_cst_cnn - np.min(smooth_cst_cnn)) / (np.max(smooth_cst_cnn) - np.min(smooth_cst_cnn))
            # plot the smooth_cst_bss and smooth_cst_cnn
            plt.subplot(2, 1, 1)
            plt.plot(smooth_cst_bss_minmax, label="Smooth CST by BSS")
            plt.plot(smooth_cst_cnn_minmax, label="Smooth CST by CNN")
            plt.title("Smooth CST by BSS (MinMax Scaled)")
            plt.subplot(2, 1, 2)
            plt.plot(smooth_cst_bss, label="Smooth CST by BSS")
            plt.plot(smooth_cst_cnn, label="Smooth CST by CNN")
            plt.title("Smooth CST by CNN")
            plt.legend()
            plt.savefig(os.path.join(save_folder, f"{cst_name}.png"))
            plt.close()
            inference_result.append({
                "model_type": modelType,
                "mix": mix,
                "repetition": repetition,
                "fold": fold,
                "dataset_type": datasetType,
                "session": session,
                "subject": subject,
                "intensity": intensity,
                "muscle": muscle,
                "segment": seg,
                "r": corr_coeff,
                "rmse": rmse,
                "f1": f1,
                "accuracy": accuracy,
            })
            printer(f"Inference result for {fileName} is:")
            printer(f"R: {corr_coeff:.3f}, RMSE: {rmse:.3f}, F1: {f1:.3f}, Accuracy: {accuracy:.3f}")
            printer(f"Saved the CST to {cst_path}")

inference_result_df = pd.DataFrame(inference_result)
inference_result_df.to_csv(os.path.join(save_dir, "inference_result.csv"), index=False)
printer(f"Saved the inference result to {os.path.join(save_dir, 'inference_result.csv')}")
