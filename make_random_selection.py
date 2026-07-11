from dataset_preparer import make_random_selection, generate_selection_table

dataDir = r"E:\NMI_Dataset\NMI_Journal_Dataset\organized\Train"
selectionTable = generate_selection_table(dataDir)
numbers_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
times = 10
saveDir = r"C:\Users\wenlab_pc_user\Dropbox\1_NMI-Projects\4_JNE_Works\Training_Validation\Random_Mix_Selection"
fname = fr"RandomSelection_Mix{numbers_list[0]}To{numbers_list[-1]}_Times{times}.json"
random_selection = make_random_selection(list(selectionTable.keys()), numbers_list, times, saveDir=saveDir, fname=fname)
for key, value in random_selection.items():
    print(f"Mix {key}")
    for selection, subset in value.items():
        print(f"Selection {selection}")
        for dataset in subset["selected_train_dataset"]:
            print(f"Selected Train Dataset: {dataset}")
        for dataset in subset["leftover_dataset"]:
            print(f"Leftover Dataset: {dataset}")
        print("--------------------------------")
    print("--------------------------------")