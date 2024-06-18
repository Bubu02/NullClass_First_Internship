import os
import shutil
import numpy as np
from tqdm import tqdm

# Define your dataset directory
dataset_dir = 'Machine Learning/ Animal Identification/Animal Dataset/Dataset'

# Define the directories for your training and testing sets
train_dir = 'Machine Learning/Animal Identification/Animal Dataset/test'
test_dir = 'Machine Learning/Animal Identification/Animal Dataset/train'

# Define the split ratio for train and test
split_ratio = 0.8  # 80% for training, 20% for testing

# Get the list of all folders in the dataset directory
folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]

# Loop over each folder
for folder in tqdm(folders, desc="Processing folders"):
    # Get the full path of the folder
    folder_path = os.path.join(dataset_dir, folder)
    
    # Get the list of all files in the folder
    files = os.listdir(folder_path)
    
    # Shuffle the files
    np.random.shuffle(files)
    
    # Split the files into train and test sets
    train_files = files[:int(len(files)*split_ratio)]
    test_files = files[int(len(files)*split_ratio):]
    
    # Create the train and test directories if they do not exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Create the folder directory in the train and test directories
    train_folder_dir = os.path.join(train_dir, folder)
    test_folder_dir = os.path.join(test_dir, folder)
    os.makedirs(train_folder_dir, exist_ok=True)
    os.makedirs(test_folder_dir, exist_ok=True)
    
    # Copy the train files into the train directory
    for file in tqdm(train_files, desc=f"Copying train files for {folder}"):
        src_file = os.path.join(folder_path, file)
        dst_file = os.path.join(train_folder_dir, file)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)
    
    # Copy the test files into the test directory
    for file in tqdm(test_files, desc=f"Copying test files for {folder}"):
        src_file = os.path.join(folder_path, file)
        dst_file = os.path.join(test_folder_dir, file)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)