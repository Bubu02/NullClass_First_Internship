import os
import cv2
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize data augmentation
data_augmentation = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))  # Resize to desired size
    img = img / 255.0  # Normalize pixel values
    img = img.reshape((1,) + img.shape + (1,))
    img = data_augmentation.flow(img, batch_size=1)[0].reshape((48, 48))
    return img

def copy_train_images(source_folder, target_folder):
    for region in range(5):
        region_folder = f'region-{region}'
        os.makedirs(os.path.join(target_folder, region_folder), exist_ok=True)
        for filename in tqdm(os.listdir(os.path.join(source_folder, f'train_{region_folder}')),
                             desc=f"Processing {region_folder} (train)"):
            input_path = os.path.join(source_folder, f'train_{region_folder}', filename)
            output_path = os.path.join(target_folder, region_folder, filename)
            preprocessed_img = preprocess_image(input_path)
            cv2.imwrite(output_path, preprocessed_img * 255)

def copy_test_images(source_folder, target_folder):
    for region in range(5):
        region_folder = f'region-{region}'
        os.makedirs(os.path.join(target_folder, region_folder), exist_ok=True)
        for filename in tqdm(os.listdir(os.path.join(source_folder, f'test_{region_folder}')),
                             desc=f"Processing {region_folder} (test)"):
            input_path = os.path.join(source_folder, f'test_{region_folder}', filename)
            output_path = os.path.join(target_folder, region_folder, filename)
            preprocessed_img = preprocess_image(input_path)
            cv2.imwrite(output_path, preprocessed_img * 255)




# Specify paths
train_folder = 'Dataset/train'
test_folder = 'Dataset/test'
output_folder = 'Dataset/Preprocessed_data'

# Preprocess train images
copy_train_images(train_folder, os.path.join(output_folder, 'train'))

# Preprocess test images
copy_test_images(test_folder, os.path.join(output_folder, 'test'))

print("Preprocessing and organization completed!")
