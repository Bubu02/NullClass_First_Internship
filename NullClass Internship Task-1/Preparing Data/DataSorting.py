import os
import shutil

# Set your dataset directory path
dataset_dir = "RegionalDataset"

# Create folders for each region
regions = ["sorted_data/region-0", "sorted_data/region-1", "sorted_data/region-2", "sorted_data/region-3", "sorted_data/region-4"]

# Get the current working directory
current_dir = os.getcwd()

for filename in os.listdir(dataset_dir):
    if filename.endswith(".jpg.chip.jpg"):
        # Extract region number
        parts = filename.split("_")
        try:
            region_number = int(parts[2]) 
        except ValueError:
            print(f"Skipping invalid filename: {filename}")
            continue

        # Check if the region number is within the valid range
        if 0 <= region_number < len(regions):
            target_folder = os.path.join(current_dir, regions[region_number])
            os.makedirs(target_folder, exist_ok=True)

            # Copy the file to the corresponding folder (instead of moving)
            source_path = os.path.join(dataset_dir, filename)
            target_path = os.path.join(target_folder, filename)

            try:
                shutil.copy(source_path, target_path)
                print(f"Copied {filename} to {target_folder}")
            except FileNotFoundError:
                print(f"Error: {filename} not found or destination folder does not exist.")
        else:
            print(f"Invalid region number {region_number} for file {filename}")

print("Images have been separated and stored in respective folders.")
