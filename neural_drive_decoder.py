import torch
import torch.nn as nn
import numpy as np

class NeuralInterface_3D(nn.Module):
    def __init__(self, classes, frameSize=(1, 40, 13, 5), numNodes=[32, 32, 32, 16, 256]):
        super(NeuralInterface_3D, self).__init__()
        self.numNodes = numNodes
        self.num_outputs = classes
        self.in_channels = frameSize[0]
        self.classes = classes
        self.trained = False

        cnnblock1 = nn.Sequential(
            nn.Conv3d(self.in_channels, numNodes[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(numNodes[0]),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Conv3d(numNodes[0], numNodes[1], kernel_size=3, padding=1),
            nn.BatchNorm3d(numNodes[1]),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.Dropout(p=0.5)
        )
        cnnblock2 = nn.Sequential(
            nn.Conv3d(numNodes[1], numNodes[2], kernel_size=3, padding=1),
            nn.BatchNorm3d(numNodes[2]),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Conv3d(numNodes[2], numNodes[3], kernel_size=3, padding=1),
            nn.BatchNorm3d(numNodes[3]),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.Dropout(p=0.5),
            nn.Flatten()
        )
        self.feature_extractor = nn.Sequential(cnnblock1, cnnblock2)
        dummy_x = torch.randn(1, *frameSize)
        dummy_feature = self.feature_extractor(dummy_x)
        flatten_shape = dummy_feature.shape
        n_features = flatten_shape[1]

        self.output = nn.ModuleList()

        for _ in range(self.classes):
            self.output.append(nn.Sequential(
                nn.Linear(n_features, numNodes[4]),
                # nn.LeakyReLU(),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(numNodes[4], 1),
                nn.Sigmoid()
            ))

    def forward(self, x):
        deep_features = self.feature_extractor(x)
        outputlist = []
        for i in range(self.num_outputs):
            outputlist.append(self.output[i](deep_features))
        output = torch.cat(outputlist, dim=1)
        return output

class NeuralInterface_2D(nn.Module):
    def __init__(self, classes, frameSize=(40, 13, 5), numNodes=[64, 64, 64, 32, 256]):
        super(NeuralInterface_2D, self).__init__()
        self.numNodes = numNodes
        self.num_outputs = classes
        self.in_channels = frameSize[0]
        self.classes = classes
        self.trained = False
        cnnblock1 = nn.Sequential(
            nn.Conv2d(self.in_channels, numNodes[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(numNodes[0]),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Conv2d(numNodes[0], numNodes[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(numNodes[1]),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.5)
        )
        cnnblock2 = nn.Sequential(
            nn.Conv2d(numNodes[1], numNodes[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(numNodes[2]),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Conv2d(numNodes[2], numNodes[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(numNodes[3]),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.5),
            nn.Flatten()
        )
        self.feature_extractor = nn.Sequential(cnnblock1, cnnblock2)
        dummy_x = torch.randn(1, *frameSize)
        dummy_feature = self.feature_extractor(dummy_x)
        flatten_shape = dummy_feature.shape
        n_features = flatten_shape[1]

        self.output = nn.ModuleList()

        for _ in range(self.classes):
            self.output.append(nn.Sequential(
                nn.Linear(n_features, numNodes[4]),
                # nn.LeakyReLU(),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(numNodes[4], 1),
                nn.Sigmoid()
            ))

    def forward(self, x):
        deep_features = self.feature_extractor(x)
        outputlist = []
        for i in range(self.num_outputs):
            outputlist.append(self.output[i](deep_features))
        output = torch.cat(outputlist, dim=1)
        return output


class NeuralInterface_1D(nn.Module):
    def __init__(self, numChannels=64, classes=4, winLen=40, numNodes=[128, 128, 128, 64, 256]):
        super(NeuralInterface_1D, self).__init__()
        self.numNodes = numNodes
        self.classes = classes
        self.channels = numChannels
        self.num_output = classes
        self.trained = False
        cnnBlock1 = nn.Sequential(nn.Conv1d(in_channels=numChannels, out_channels=numNodes[0], kernel_size=3),
                                  nn.BatchNorm1d(numNodes[0]),
                                  nn.ReLU(),
                                  nn.Conv1d(in_channels=numNodes[0], out_channels=numNodes[1], kernel_size=3),
                                  nn.BatchNorm1d(numNodes[1]),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=2),
                                  nn.Dropout(p=0.5))
        cnnBlock2 = nn.Sequential(nn.Conv1d(in_channels=numNodes[1], out_channels=numNodes[2], kernel_size=3),
                                  nn.BatchNorm1d(numNodes[2]),
                                  nn.ReLU(),
                                  nn.Conv1d(in_channels=numNodes[2], out_channels=numNodes[3], kernel_size=3),
                                  nn.BatchNorm1d(numNodes[3]),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=2),
                                  nn.Dropout(p=0.5),
                                  nn.Flatten())

        self.feature_extractor = nn.Sequential(cnnBlock1, cnnBlock2)
        dummy_input = torch.randn(1, numChannels, winLen)
        dummy_output = self.feature_extractor(dummy_input)
        n_features = dummy_output.numel() // dummy_output.size(0)
        self.output = nn.ModuleList()

        for _ in range(self.classes):
            self.output.append(nn.Sequential(
                nn.Linear(n_features, numNodes[4]),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(numNodes[4], 1),
                nn.Sigmoid()
            ))

    def forward(self, x):
        deep_features = self.feature_extractor(x)
        outputlist = []
        for i in range(self.classes):
            outputlist.append(self.output[i](deep_features))
        output = torch.cat(outputlist, dim=1)
        return output      

class NeuralInterface_1D_v0(nn.Module):
    def __init__(self, numChannels=64, classes=4, winLen=40, numNodes=[128, 128, 128, 64, 256]):
        super(NeuralInterface_1D_v0, self).__init__()
        self.numNodes = numNodes
        self.classes = classes
        self.channels = numChannels
        self.num_output = classes
        cnnBlock1 = nn.Sequential(nn.Conv1d(in_channels=numChannels, out_channels=numNodes[0], kernel_size=3),
                                  nn.BatchNorm1d(numNodes[0]),
                                  nn.ReLU(),
                                  nn.Conv1d(in_channels=numNodes[0], out_channels=numNodes[1], kernel_size=3),
                                  nn.BatchNorm1d(numNodes[1]),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=2),
                                  nn.Dropout(p=0.5))
        cnnBlock2 = nn.Sequential(nn.Conv1d(in_channels=numNodes[1], out_channels=numNodes[2], kernel_size=3),
                                  nn.BatchNorm1d(numNodes[2]),
                                  nn.ReLU(),
                                  nn.Conv1d(in_channels=numNodes[2], out_channels=numNodes[3], kernel_size=3),
                                  nn.BatchNorm1d(numNodes[3]),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=2),
                                  nn.Dropout(p=0.5),
                                  nn.Flatten())

        self.feature_extractor = nn.Sequential(cnnBlock1, cnnBlock2)
        dummy_input = torch.randn(1, numChannels, winLen)
        dummy_output = self.feature_extractor(dummy_input)
        n_features = dummy_output.numel() // dummy_output.size(0)
        self.output = nn.ModuleList()

        for _ in range(self.classes):
            self.output.append(nn.Sequential(
                nn.Linear(n_features, numNodes[4]),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(numNodes[4], 1),
                nn.Sigmoid()
            ))

    def forward(self, x):
        deep_features = self.feature_extractor(x)
        outputlist = []
        for i in range(self.classes):
            outputlist.append(self.output[i](deep_features))
        output = torch.cat(outputlist, dim=1)
        return output
        


# class NeuralInterface_2D(nn.Module):
#     def __init__(self, classes, frameSize=(40, 13, 5), numNodes=[64, 64, 64, 32, 128]):
#         super(NeuralInterface_2D, self).__init__()
#         self.numNodes = numNodes
#         self.num_outputs = classes
#         self.in_channels = frameSize[0]
#         self.classes = classes

#         cnnblock1 = nn.Sequential(
#             nn.Conv2d(self.in_channels, numNodes[0], kernel_size=3, padding=3),
#             nn.BatchNorm2d(numNodes[0]),
#             nn.LeakyReLU(),
#             nn.Conv2d(numNodes[0], numNodes[1], kernel_size=3, padding=3),
#             nn.BatchNorm2d(numNodes[1]),
#             nn.LeakyReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=3),
#             nn.Dropout(p=0.5)
#         )
#         cnnblock2 = nn.Sequential(
#             nn.Conv2d(numNodes[1], numNodes[2], kernel_size=3, padding=3),
#             nn.BatchNorm2d(numNodes[2]),
#             nn.LeakyReLU(),
#             nn.Conv2d(numNodes[2], numNodes[3], kernel_size=3, padding=3),
#             nn.BatchNorm2d(numNodes[3]),
#             nn.LeakyReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=3),
#             nn.Dropout(p=0.5),
#             nn.Flatten()
#         )
#         self.feature_extractor = nn.Sequential(cnnblock1, cnnblock2)
#         dummy_x = torch.randn(1, *frameSize)
#         dummy_feature = self.feature_extractor(dummy_x)
#         flatten_shape = dummy_feature.shape
#         n_features = flatten_shape[1]

#         self.output = nn.ModuleList()

#         for _ in range(self.classes):
#             self.output.append(nn.Sequential(
#                 nn.Linear(n_features, numNodes[4]),
#                 nn.LeakyReLU(),
#                 nn.Dropout(0.5),
#                 nn.Linear(numNodes[4], 1),
#                 nn.Sigmoid()
#             ))

#     def forward(self, x):
#         deep_features = self.feature_extractor(x)
#         outputlist = []
#         for i in range(self.num_outputs):
#             outputlist.append(self.output[i](deep_features))
#         output = torch.cat(outputlist, dim=1)
#         return output


# class NeuralInterface_2D_v2(nn.Module):
#     """
#     Revised 2D CNN for a (40, 13, 5) neurophysiological input
#     (40 feature channels over a 13 x 5 spatial/temporal map).

#     Changes vs. the original:
#       - Conv padding 3 -> 1 ("same" for kernel_size=3). The old setting
#         inflated each feature map with zero borders (13->17->21), diluting
#         the real signal and skewing BatchNorm statistics. Now the convs
#         preserve spatial size and always straddle real data.
#       - MaxPool kernel/stride 3 -> 2. With "same" convs, k3/s3 pooling would
#         collapse the width-5 axis to 0; k2/s2 is the appropriate, gentler
#         downsampling for such a small input.
#       - Conv stride stays 1; all downsampling happens in the pools.

#     Spatial trace (per sample):
#       (40,13,5)
#         block1: conv,conv (same) -> (64,13,5) -> pool2 -> (64,6,2)
#         block2: conv,conv (same) -> (32,6,2)  -> pool2 -> (32,3,1)
#         flatten -> 96 features
#     """

#     def __init__(self, classes, frameSize=(40, 13, 5), numNodes=[64, 64, 64, 32, 128]):
#         super(NeuralInterface_2D_v2, self).__init__()
#         self.numNodes = numNodes
#         self.num_outputs = classes
#         self.in_channels = frameSize[0]
#         self.classes = classes

#         cnnblock1 = nn.Sequential(
#             nn.Conv2d(self.in_channels, numNodes[0], kernel_size=3, padding=1),
#             nn.BatchNorm2d(numNodes[0]),
#             nn.LeakyReLU(),
#             # nn.ELU(),
#             # nn.ReLU(),
#             nn.Conv2d(numNodes[0], numNodes[1], kernel_size=3, padding=1),
#             nn.BatchNorm2d(numNodes[1]),
#             nn.LeakyReLU(),
#             # nn.ELU(),
#             # nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(p=0.5)
#             # If the width-5 axis is meaningful and you want to keep all 5
#             # columns, swap the pool above for height-only pooling:
#             #   nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
#             # and do the same in block2. The dummy pass below re-sizes the
#             # classifier automatically either way.
#         )
#         cnnblock2 = nn.Sequential(
#             nn.Conv2d(numNodes[1], numNodes[2], kernel_size=3, padding=1),
#             nn.BatchNorm2d(numNodes[2]),
#             nn.LeakyReLU(),
#             # nn.ELU(),
#             # nn.ReLU(),
#             nn.Conv2d(numNodes[2], numNodes[3], kernel_size=3, padding=1),
#             nn.BatchNorm2d(numNodes[3]),
#             nn.LeakyReLU(),
#             # nn.ELU(),
#             # nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(p=0.5),
#             nn.Flatten()
#         )
#         self.feature_extractor = nn.Sequential(cnnblock1, cnnblock2)
#         dummy_x = torch.randn(1, *frameSize)
#         dummy_feature = self.feature_extractor(dummy_x)
#         flatten_shape = dummy_feature.shape
#         n_features = flatten_shape[1]

#         self.output = nn.ModuleList()

#         for _ in range(self.classes):
#             self.output.append(nn.Sequential(
#                 nn.Linear(n_features, numNodes[4]),
#                 nn.ReLU(),
#                 # nn.LeakyReLU(),
#                 nn.Dropout(0.5),
#                 nn.Linear(numNodes[4], 1),
#                 nn.Sigmoid()
#             ))

#     def forward(self, x):
#         deep_features = self.feature_extractor(x)
#         outputlist = []
#         for i in range(self.num_outputs):
#             outputlist.append(self.output[i](deep_features))
#         output = torch.cat(outputlist, dim=1)
#         return output


# class NeuralInterface_3D(nn.Module):
#     def __init__(self, classes, frameSize=(1, 40, 13, 5), numNodes=[64, 64, 64, 32, 128]):
#         super(NeuralInterface_3D, self).__init__()
#         self.numNodes = numNodes
#         self.num_outputs = classes
#         self.in_channels = frameSize[0]
#         self.classes = classes

#         cnnblock1 = nn.Sequential(
#             nn.Conv3d(self.in_channels, numNodes[0], kernel_size=3, padding=2),
#             nn.BatchNorm3d(numNodes[0]),
#             # nn.LeakyReLU(),
#             nn.ReLU(),
#             nn.Conv3d(numNodes[0], numNodes[1], kernel_size=3, padding=2),
#             nn.BatchNorm3d(numNodes[1]),
#             # nn.LeakyReLU(),
#             nn.ReLU(),
#             nn.MaxPool3d(kernel_size=3, stride=3),
#             nn.Dropout(p=0.5)
#         )
#         cnnblock2 = nn.Sequential(
#             nn.Conv3d(numNodes[1], numNodes[2], kernel_size=3, padding=2),
#             nn.BatchNorm3d(numNodes[2]),
#             # nn.LeakyReLU(),
#             nn.ReLU(),
#             nn.Conv3d(numNodes[2], numNodes[3], kernel_size=3, padding=2),
#             nn.BatchNorm3d(numNodes[3]),
#             # nn.LeakyReLU(),
#             nn.ReLU(),
#             nn.MaxPool3d(kernel_size=3, stride=3),
#             nn.Dropout(p=0.5),
#             nn.Flatten()
#         )
#         self.feature_extractor = nn.Sequential(cnnblock1, cnnblock2)
#         dummy_x = torch.randn(1, *frameSize)
#         dummy_feature = self.feature_extractor(dummy_x)
#         flatten_shape = dummy_feature.shape
#         n_features = flatten_shape[1]

#         self.output = nn.ModuleList()

#         for _ in range(self.classes):
#             self.output.append(nn.Sequential(
#                 nn.Linear(n_features, numNodes[4]),
#                 # nn.LeakyReLU(),
#                 nn.ReLU(),
#                 nn.Dropout(0.5),
#                 nn.Linear(numNodes[4], 1),
#                 nn.Sigmoid()
#             ))

#     def forward(self, x):
#         deep_features = self.feature_extractor(x)
#         outputlist = []
#         for i in range(self.num_outputs):
#             outputlist.append(self.output[i](deep_features))
#         output = torch.cat(outputlist, dim=1)
#         return output


# class NeuralInterface_3D_v2(nn.Module):
#     """
#     Revised 3D CNN for a (1, 40, 13, 5) neurophysiological input
#     (1 input channel over a 40 x 13 x 5 volume).

#     Changes vs. the original (architecture unchanged):

#       Parametric layers
#         - Conv3d padding 2 -> 1 ("same" for kernel_size=3). The old setting
#           inflated the volume on every axis (40->42->44, 13->15->17, 5->7->9),
#           which in 3D costs compute/memory cubically, dilutes the real signal,
#           and feeds zero-heavy maps into BatchNorm. Now the convs preserve
#           volume and always straddle real data. Conv stride stays 1.
#         - BatchNorm3d momentum 0.1 -> 0.05 for more stable running stats with
#           the small batch sizes typical of neuro datasets.
#         - Explicit Kaiming/He init matched to the LeakyReLU slope (PyTorch's
#           default a=sqrt(5) is not matched to LeakyReLU).

#       Nonparametric layers
#         - MaxPool3d kernel/stride 3 -> 2. With "same" convs, k3/s3 would
#           over-shrink the small H=13 / W=5 axes; k2/s2 is the right scale.
#         - Dropout lowered to 0.3 in the conv blocks (0.5 was aggressive right
#           after convolutions); the FC head keeps 0.5.

#     Volume trace (per sample):
#       (1,40,13,5)
#         block1: conv,conv (same) -> (64,40,13,5) -> pool2 -> (64,20,6,2)
#         block2: conv,conv (same) -> (32,20,6,2)  -> pool2 -> (32,10,3,1)
#         flatten -> 960 features
#     """

#     def __init__(self, classes, frameSize=(1, 40, 13, 5), numNodes=[64, 64, 64, 32, 128]):
#         super(NeuralInterface_3D_v2, self).__init__()
#         self.numNodes = numNodes
#         self.num_outputs = classes
#         self.in_channels = frameSize[0]
#         self.classes = classes

#         cnnblock1 = nn.Sequential(
#             nn.Conv3d(self.in_channels, numNodes[0], kernel_size=3, padding=1),
#             nn.BatchNorm3d(numNodes[0], momentum=0.05),
#             nn.LeakyReLU(),
#             nn.Conv3d(numNodes[0], numNodes[1], kernel_size=3, padding=1),
#             nn.BatchNorm3d(numNodes[1], momentum=0.05),
#             nn.LeakyReLU(),
#             nn.MaxPool3d(kernel_size=2, stride=2),
#             nn.Dropout(p=0.3)
#             # If the 40-axis is UNORDERED channels (not a physically ordered
#             # electrode axis), prefer kernel_size=(1, 3, 3) on the convs above
#             # so weights aren't shared across unrelated channels, and/or pool
#             # anisotropically with kernel_size=(2, 2, 1) to keep all 5 width
#             # columns. The dummy pass below re-sizes the classifier either way.
#         )
#         cnnblock2 = nn.Sequential(
#             nn.Conv3d(numNodes[1], numNodes[2], kernel_size=3, padding=1),
#             nn.BatchNorm3d(numNodes[2], momentum=0.05),
#             nn.LeakyReLU(),
#             nn.Conv3d(numNodes[2], numNodes[3], kernel_size=3, padding=1),
#             nn.BatchNorm3d(numNodes[3], momentum=0.05),
#             nn.LeakyReLU(),
#             nn.MaxPool3d(kernel_size=2, stride=2),
#             nn.Dropout(p=0.3),
#             nn.Flatten()
#         )
#         self.feature_extractor = nn.Sequential(cnnblock1, cnnblock2)
#         dummy_x = torch.randn(1, *frameSize)
#         dummy_feature = self.feature_extractor(dummy_x)
#         flatten_shape = dummy_feature.shape
#         n_features = flatten_shape[1]

#         self.output = nn.ModuleList()

#         for _ in range(self.classes):
#             self.output.append(nn.Sequential(
#                 nn.Linear(n_features, numNodes[4]),
#                 nn.LeakyReLU(),
#                 nn.Dropout(0.5),
#                 nn.Linear(numNodes[4], 1),
#                 nn.Sigmoid()
#             ))

#         self._init_weights()

#     def _init_weights(self):
#         # Kaiming/He init matched to LeakyReLU (default negative_slope = 0.01).
#         for m in self.modules():
#             if isinstance(m, (nn.Conv3d, nn.Linear)):
#                 nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity='leaky_relu')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def forward(self, x):
#         deep_features = self.feature_extractor(x)
#         outputlist = []
#         for i in range(self.num_outputs):
#             outputlist.append(self.output[i](deep_features))
#         output = torch.cat(outputlist, dim=1)
#         return output