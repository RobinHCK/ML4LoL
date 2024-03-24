import pandas as pd
# StratifiedKFold to preserve same size for each split
from sklearn.model_selection import StratifiedKFold 
import json
import os
from utils.tools import showCurrentTime
from utils.constants import championsList
from utils.constants import champion_id_map
import numpy as np

import shutil

from config import pro_patch, pro, ranked, patch

# split dataset using the patch used and the type of dataset
def split_dataset(dataset_type:str):

    if dataset_type == "ranked":
        input_path = "dataset/"+dataset_type+"_dataset_"+patch+".csv"
    elif dataset_type == "pro":
        if pro_patch==None:
            print(showCurrentTime() + "Pro dataset patch is not given in config file !")
            return
        input_path = "dataset/"+dataset_type+"_dataset_"+pro_patch+".csv"
    else :
        print(showCurrentTime() + "CAREFUL THE DATASET TYPE IS NOT VALID !")
        return
        
    dataset = pd.read_csv(input_path)

    # Create a folder to store the split files if it doesn't exist
    outputFolder = 'data'
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
        
    # Create feature vectors for the blue and red team
    blue_features = dataset[['blueTopChampion', 'blueJungleChampion', 'blueMidChampion', 'blueCarryChampion', 'blueSupportChampion']]
    blue_features = blue_features.apply(lambda col: col.map(champion_id_map), axis=1)

    red_features = dataset[['redTopChampion', 'redJungleChampion', 'redMidChampion', 'redCarryChampion', 'redSupportChampion']]
    red_features = red_features.apply(lambda col: col.map(champion_id_map), axis=1)

    # Create a target vector
    # blue is 1, else (red) it's 0
    target = dataset['winner'].apply(lambda x: 1 if x == 'Blue Team' else 0)

    # Combine the feature vectors for both teams
    team_features = pd.concat([blue_features, red_features], axis=1)

    # number of splits for cross-validation
    n_splits = 5
    # Calculate the number of rows per subset
    rows_per_subset = len(team_features) // n_splits

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Split the data into subsets
    subsets=[]
    for train_index, test_index in skf.split(team_features, target):
        subset = team_features.iloc[test_index]
        subsets.append(subset)

    for i, subset in enumerate(subsets):
        # Get the corresponding indices from the original dataset
        indice = subset.index

        # Create subsets using the indices
        subset_data = dataset.loc[indice]

        # Save subsets to files or use them as needed
        if dataset_type != None:
            subset_data.to_csv(os.path.join(outputFolder, f'{dataset_type}_fold{i+1}.csv'), index=False)

        else:
            subset_data.to_csv(os.path.join(outputFolder, f'fold{i+1}.csv'), index=False)

    # list containing files data + files names
    files = []
    filesNames = []
    # from fold 1 to split N, we fill the lists
    for i in range(1,n_splits+1):
        if dataset_type:
            file_name = 'data/'+dataset_type+'_fold'+str(i)+'.csv'
            file = pd.read_csv(file_name)
            files.append(file)
            filesNames.append(file_name)
        else :
            file_name = 'data/fold'+str(i)+'.csv'
            file = pd.read_csv(file_name)
            files.append(file)
            filesNames.append(file_name)

    ###########################################################
    # Make sure each file has exactly the same amount of rows #
    ###########################################################

    # we get the file with the lowest amount of rows as reference
    min_rows = min(len(file) for file in files)

    # we replace the csv files by taking only the min_rows first rows (remove additional rows)
    for i in range(1,n_splits+1):
        file = files[i-1].head(min_rows)
        file.to_csv(filesNames[i-1], index=False)

    # in each fold folder, create a train folder containing the others folds
    for i in range(1,n_splits+1):
        # Create the second-level folder (train) inside each fold
        if dataset_type != None:

            test_folder = 'data/'+dataset_type+'/fold'+str(i)+'/test'
            os.makedirs(test_folder, exist_ok=True)
            test_file_path = os.path.join(test_folder, "raw_data.csv")
            shutil.copy('data/'+dataset_type+'_fold'+str(i)+'.csv', test_file_path)

            train_folder = 'data/'+dataset_type+'/fold'+str(i)+'/train'
            os.makedirs(train_folder, exist_ok=True)

            # Initialize an empty DataFrame to concatenate all CSV files
            combined_df = pd.DataFrame()

            for j in range(1,n_splits+1):

                if j!=i:
                    source_path = 'data/'+dataset_type+'_fold'+str(j)+'.csv'

                    # Read the CSV file and concatenate it to the combined DataFrame
                    df = pd.read_csv(source_path)
                    combined_df = pd.concat([combined_df, df])

            # Save the combined DataFrame to the train folder
            destination_path = os.path.join(train_folder, 'raw_data.csv')
            combined_df.to_csv(destination_path, index=False)

        else:

            train_folder = os.path.join('data', 'train')
            os.makedirs(train_folder, exist_ok=True)

            # Initialize an empty DataFrame to concatenate all CSV files
            combined_df = pd.DataFrame()

            for j in range(1,n_splits+1):

                if j!=i:
                    source_path = 'data/fold'+str(i)+'.csv'
                    # Read the CSV file and concatenate it to the combined DataFrame
                    df = pd.read_csv(source_path)
                    combined_df = pd.concat([combined_df, df])

            # Save the combined DataFrame to the train folder
            destination_path = os.path.join(train_folder, 'raw_data.csv')
            combined_df.to_csv(destination_path, index=False)

    ########################################################################
    # Make sure each file is unique (no duplicates from 1 file to another) #
    ########################################################################

    # Concatenate all files into one DataFrame
    all_files = pd.concat(file for file in files)

    # Identify duplicate rows
    duplicates = all_files[all_files.duplicated()]

    if not duplicates.empty:
        print(showCurrentTime() + "Duplicate rows found (it may be because some compositions were already played):")
        print(duplicates)
    else:
        print(showCurrentTime() + "No duplicate rows found.")

###########################################################
# SPLIT DATASET GIVEN BY GIVING THE TYPE OF DATASET HANDLED
###########################################################
print("\n----")
print(showCurrentTime() + "SPLITING RANKED DATASET:")
print("----")
split_dataset("ranked")

print("\n----")
print(showCurrentTime() + "SPLITING PRO DATASET:")
print("----")
split_dataset("pro")