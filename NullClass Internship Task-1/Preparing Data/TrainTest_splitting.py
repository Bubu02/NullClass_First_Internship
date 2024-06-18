import os
import shutil
from sklearn.model_selection import train_test_split

# Path to your dataset folder
dataset_folder = "sorted_data"

# Get the current working directory
current_dir = os.getcwd()

# Create train and test folders
train_folder = os.path.join(current_dir, "train")
test_folder = os.path.join(current_dir, "test")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Get a list of region folders
region_folders = [folder for folder in os.listdir(dataset_folder) if folder.startswith("region-")]

# Split data and move to train and test folders
for region_folder in region_folders:
    region_path = os.path.join(dataset_folder, region_folder)
    images = os.listdir(region_path)
    train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

    # Create subfolders in train and test
    train_subfolder = os.path.join(train_folder, f"train_{region_folder}")
    test_subfolder = os.path.join(test_folder, f"test_{region_folder}")
    os.makedirs(train_subfolder, exist_ok=True)
    os.makedirs(test_subfolder, exist_ok=True)

    # Move images to respective subfolders
    for image in train_images:
        shutil.copy(os.path.join(region_path, image), os.path.join(train_subfolder, image))
    for image in test_images:
        shutil.copy(os.path.join(region_path, image), os.path.join(test_subfolder, image))

print("Data split and folders created successfully!")
