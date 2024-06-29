import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import filedialog, Button
from keras.losses import MeanAbsoluteError
from PIL import Image, ImageTk
import tkinter as tk
import torch

# Loading the trained model for age detection
custom_objects = {
    'mae': MeanAbsoluteError()
}

# Load your pre-trained models
sleep_model = load_model('Machine Learning/Sleep Detection/Saved Models/Car_Drowsiness_Detection_Model_v2.0.h5')
age_model = load_model('Machine Learning/Age Detector/Saved Models/Age_Detection_Model_v1.1.h5', custom_objects=custom_objects)

# Load YOLOv5 model with custom weights
yolo_model_path = 'Main GUI/yolov5s.pt'  # Update this path to your .pt file
model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_model_path, force_reload=True)

def preprocess_image(image, target_size=(224, 224), grayscale=False):
    # Resize image to target size and normalize
    img = cv2.resize(image, target_size)
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape((img.shape[0], img.shape[1], 1))  # Add channel dimension for grayscale
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def detect_people(image):
    results = model(image)
    detections = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, conf, cls
    people_detections = [d for d in detections if int(d[5]) == 0]  # filter 'person' class
    return people_detections

def classify_sleep(person_crop):
    processed_crop = preprocess_image(person_crop, target_size=(224, 224), grayscale=False)
    sleep_pred = sleep_model.predict(processed_crop)
    return sleep_pred[0][0] > 0.5  # Binary classification: sleeping or not

def predict_age(person_crop):
    processed_crop = preprocess_image(person_crop, target_size=(128, 128), grayscale=True)
    age_pred = age_model.predict(processed_crop)
    return round(age_pred[0][0])

def process_frame(frame):
    people_detections = detect_people(frame)
    sleeping_count = 0
    results = []

    for detection in people_detections:
        x1, y1, x2, y2, conf, cls = detection
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        person_crop = frame[y1:y2, x1:x2]

        if classify_sleep(person_crop):
            sleeping_count += 1
            age = predict_age(person_crop)
            results.append({'bbox': [x1, y1, x2, y2], 'age': age})

    return sleeping_count, results

def detect_age_and_sleep(file_path):
    frame = cv2.imread(file_path)
    sleeping_count, results = process_frame(frame)

    for result in results:
        x1, y1, x2, y2 = result['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"Age: {result['age']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(frame, "Sleeping", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    label_img.config(image=imgtk)
    label_img.image = imgtk

    label_status.config(text=f"Number of sleeping persons: {sleeping_count}")

def upload_image():
    file_path = filedialog.askopenfilename()
    detect_age_and_sleep(file_path)

# Creating the main window
root = tk.Tk()
root.title("Age and Sleep Detection")
root.geometry('800x600')  # Set initial size

# Creating a button for uploading images
button_upload = tk.Button(root, text="Upload Image", command=upload_image)
button_upload.pack()

# Creating a label for displaying the uploaded image
label_img = tk.Label(root)
label_img.pack()

# Creating a label for displaying the predicted age and sleep status
label_status = tk.Label(root, text="Predicted Age and Sleep Status: ")
label_status.pack()

# Adding a close button
button_close = Button(root, text="Close", command=root.destroy)
button_close.pack()

# Run the application
root.mainloop()
