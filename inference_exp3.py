"""
This script is to test the CNN models trained in Exp3 which was done in 2025 September.
"""
import torch
from torch import nn
import os
import numpy as np
import glob
import pandas as pd
from decoder_trainer_tools import NeuralDataset, inference_model, inference_model_simple_version
from dataset_preparer import spikes2cst, encode_cst, cnn_reshaper
from torch.utils.data import DataLoader
from utils import setup_logger, VerbosePrinter
import joblib
import gc

class NeuralInterface_1D_v2(nn.Module):
    def __init__(self, numChannels=64, classes=4, winLen=40, numNodes=[128, 128, 128, 64, 256]):
        super(NeuralInterface_1D_v2, self).__init__()
        self.numNodes = numNodes
        self.classes = classes
        self.channels = numChannels
        self.num_output = classes

        cnnBlock1 = nn.Sequential(
            nn.Conv1d(in_channels=numChannels, out_channels=numNodes[0], kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=numNodes[0], out_channels=numNodes[1], kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.5),
        )
        cnnBlock2 = nn.Sequential(
            nn.Conv1d(in_channels=numNodes[1], out_channels=numNodes[2], kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=numNodes[2], out_channels=numNodes[3], kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.5),
            nn.Flatten(),
        )

        self.feature_extractor = nn.Sequential(cnnBlock1, cnnBlock2)
        dummy_input = torch.randn(1, numChannels, winLen)
        dummy_output = self.feature_extractor(dummy_input)
        n_features = dummy_output.numel() // dummy_output.size(0)

        self.outputs = nn.ModuleList()
        for _ in range(self.classes):
            self.outputs.append(nn.Sequential(
                nn.Linear(n_features, numNodes[4]),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(numNodes[4], 1),
                nn.Sigmoid(),
            ))

    def forward(self, x):
        deep_features = self.feature_extractor(x)
        outputlist = []
        for i in range(self.classes):
            outputlist.append(self.outputs[i](deep_features))
        output = torch.cat(outputlist, dim=1)
        return output


class NeuralInterface_2D_v5(nn.Module):
    def __init__(self, classes=4, frameSize=(40, 13, 5), numNodes=[64, 64, 64, 64, 256]):
        super(NeuralInterface_2D_v5, self).__init__()
        self.numNodes = numNodes
        self.num_outputs = classes
        self.in_channels = frameSize[0]
        self.classes = classes

        self.cnnblock1 = nn.Sequential(
            nn.Conv2d(self.in_channels, numNodes[0], kernel_size=3, padding=3),
            nn.BatchNorm2d(numNodes[0]),
            nn.LeakyReLU(),
            nn.Conv2d(numNodes[0], numNodes[1], kernel_size=3, padding=3),
            nn.BatchNorm2d(numNodes[1]),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Dropout(p=0.5),
        )
        self.cnnblock2 = nn.Sequential(
            nn.Conv2d(numNodes[1], numNodes[2], kernel_size=3, padding=3),
            nn.BatchNorm2d(numNodes[2]),
            nn.LeakyReLU(),
            nn.Conv2d(numNodes[2], numNodes[3], kernel_size=3, padding=3),
            nn.BatchNorm2d(numNodes[3]),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Dropout(p=0.5),
        )

        dummy_x = torch.randn(1, *frameSize)
        dummy_feature = self.cnnblock2(self.cnnblock1(dummy_x))
        n_features = dummy_feature.numel() // dummy_feature.size(0)

        self.denses = nn.ModuleList()
        for _ in range(self.classes):
            self.denses.append(nn.Sequential(
                nn.Linear(n_features, numNodes[4]),
                nn.LeakyReLU(),
                nn.Dropout(0.5),
                nn.Linear(numNodes[4], 1),
                nn.Sigmoid(),
            ))

    def forward(self, x):
        x = self.cnnblock1(x)
        x = self.cnnblock2(x)
        deep_features = torch.flatten(x, start_dim=1)
        outputlist = []
        for i in range(self.num_outputs):
            outputlist.append(self.denses[i](deep_features))
        output = torch.cat(outputlist, dim=1)
        return output


class NeuralInterface_3D_v5(nn.Module):
    def __init__(self, classes=4, frameSize=(1, 40, 13, 5), numNodes=[32, 32, 32, 32, 256]):
        super(NeuralInterface_3D_v5, self).__init__()
        self.numNodes = numNodes
        self.num_outputs = classes
        self.in_channels = frameSize[0]
        self.classes = classes

        self.cnnblock1 = nn.Sequential(
            nn.Conv3d(self.in_channels, numNodes[0], kernel_size=3, padding=2),
            nn.BatchNorm3d(numNodes[0]),
            nn.LeakyReLU(),
            nn.Conv3d(numNodes[0], numNodes[1], kernel_size=3, padding=2),
            nn.BatchNorm3d(numNodes[1]),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=3, stride=3),
            nn.Dropout(p=0.5),
        )
        self.cnnblock2 = nn.Sequential(
            nn.Conv3d(numNodes[1], numNodes[2], kernel_size=3, padding=2),
            nn.BatchNorm3d(numNodes[2]),
            nn.LeakyReLU(),
            nn.Conv3d(numNodes[2], numNodes[3], kernel_size=3, padding=2),
            nn.BatchNorm3d(numNodes[3]),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=3, stride=3),
            nn.Dropout(p=0.5),
        )

        dummy_x = torch.randn(1, *frameSize)
        dummy_feature = self.cnnblock2(self.cnnblock1(dummy_x))
        n_features = dummy_feature.numel() // dummy_feature.size(0)

        self.outputs = nn.ModuleList()
        for _ in range(self.classes):
            self.outputs.append(nn.Sequential(
                nn.Linear(n_features, numNodes[4]),
                nn.LeakyReLU(),
                nn.Dropout(0.5),
                nn.Linear(numNodes[4], 1),
                nn.Sigmoid(),
            ))

    def forward(self, x):
        x = self.cnnblock1(x)
        x = self.cnnblock2(x)
        deep_features = torch.flatten(x, start_dim=1)
        outputlist = []
        for i in range(self.num_outputs):
            outputlist.append(self.outputs[i](deep_features))
        output = torch.cat(outputlist, dim=1)
        return output

def main(result_path, save_path, target_mix, inference_dataset):
    logger = setup_logger("Inference", log_dir=save_path)
    printer = VerbosePrinter(logger)
    printer("Start inference")
    SG = 3
    # search for pth files recursively in the result_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    printer(f"Will use the device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}for inference")
    pth_files_list = glob.glob(os.path.join(result_path, "**/*.pth"), recursive=True)
    inference_data_list = glob.glob(os.path.join(inference_dataset, "**/*.pkl"), recursive=True)
    for pth_file in pth_files_list:
        fileName = os.path.basename(pth_file)
        fileDir = os.path.normpath(os.path.dirname(pth_file))
        fileDirParts = fileDir.split(os.sep)
        fold = fileDirParts[-1][3:]
        repetition = fileDirParts[-2][5:]
        mixParts = int(fileDirParts[-3].split("_")[0][2:])
        modelType = int(fileDirParts[-4].split("_")[1][0])
        if mixParts == target_mix:
            printer(f"Processing {fileName} with mix {mixParts} and model type {modelType}")
            if modelType == 1:
                model = NeuralInterface_1D_v2()

            elif modelType == 2:
                model = NeuralInterface_2D_v5()
            elif modelType == 3:
                model = NeuralInterface_3D_v5()
            else:
                raise ValueError(f"Invalid model type: {model}")
            model.load_state_dict(torch.load(pth_file))
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
                    cst = spikes2cst(spikes)
                    cst = encode_cst(cst, num_outputs=4)
                    emg_reshaped = cnn_reshaper(emg, modelType)
                    printer(f"The shape of the loaded EMG is {emg.shape}, loaded spikes is {spikes.shape}, reshaped EMG is {emg_reshaped.shape}, encoded CST is {cst.shape}")
                    test_dataset = NeuralDataset(emg_reshaped, cst)
                    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                    pred_cst, bss_cst, _, _ = inference_model_simple_version(model, test_loader, device, intensity)
                    cst_name = f"CNN-{modelType}_N-{mixParts}_R-{repetition}_F-{fold}_T-{datasetType}_D-{session}_S-{subject}_I-{intensity}_M-{muscle}_SG-{seg}.csv"
                    local_result_df = pd.DataFrame({
                        "CST_BSS": np.asarray(bss_cst).ravel().astype(np.float64),
                        "CST_CNN": np.asarray(pred_cst).ravel().astype(np.float64),
                    })
                    local_result_df.to_csv(os.path.join(save_path, cst_name), index=False)
                    printer(f"Saved the result to {os.path.join(save_path, cst_name)}")
            del model
            torch.cuda.empty_cache()
            gc.collect()
        else:
            printer(f"Skipping {fileName} because the mix is not {target_mix}")

if __name__ == "__main__":
    result_path = r"C:\Users\wenlab_pc_user\Dropbox\1_NMI-Projects\1_Training-Results\2025_Mid_Train_Result_Old_Protocol\Exp3"
    result_save_path = r"C:\Users\wenlab_pc_user\Dropbox\1_NMI-Projects\5_JNE_Revision_Code_Works\NMI_JNE_EXP3_Inference_Validation\Inference_Raw"
    os.makedirs(result_save_path, exist_ok=True)
    target_mix = 11
    inference_dataset = r"C:\Users\wenlab_pc_user\Documents\nmi_inference\Evaluation"

    main(result_path, result_save_path, target_mix, inference_dataset)
