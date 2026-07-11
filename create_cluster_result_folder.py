import os

TASK_NAME = "Exp7"
MODEL_TYPE = ["1D", "2D", "3D"]
MIX_NUMBERS = [i for i in range(1, 20)]
REPEAT_TIMES = 10

targetFolder = r"C:\Users\wenlab_pc_user\Dropbox\1_NMI-Projects\4_JNE_Works\Training_Validation\Random_Mix_Selection"

for modelType in MODEL_TYPE:
    for mixNumber in MIX_NUMBERS:
        folderName = f"Work-{TASK_NAME}_Model-{modelType}_Mix-{mixNumber}_Rep-{REPEAT_TIMES}"
        folderPath = os.path.join(targetFolder, TASK_NAME, folderName)
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
            print(f"Created folder: {folderPath}")
        else:
            print(f"Folder already exists: {folderPath}")