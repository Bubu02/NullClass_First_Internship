import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import numpy as np
import cv2
import torch
from keras.losses import MeanAbsoluteError

# Load the trained models
custom_objects = {
    'mae': MeanAbsoluteError()
}

sleep_model = load_model('Machine Learning/Sleep Detection/Saved Models/Car_Drowsiness_Detection_Model_v2.0.h5')
age_model = load_model('Machine Learning/Age Detector/Saved Models/Age_Detection_Model_v1.1.h5', custom_objects=custom_objects)

# Load YOLOv5 model with custom weights
yolo_model_path = 'Main GUI/yolov5s.pt'  # Update this path to your .pt file
model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_model_path, force_reload=True)

# Load Haar Cascade for face detection
haarcascade_path = 'Main GUI/haarcascade_frontalface_default.xml'  # Ensure this file is downloaded and available
face_cascade = cv2.CascadeClassifier(haarcascade_path)

def preprocess_image(image, target_size=(224, 224), grayscale=False):
    img = cv2.resize(image, target_size)
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape((img.shape[0], img.shape[1], 1))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def detect_people(image):
    results = model(image)
    detections = results.xyxy[0].cpu().numpy()
    people_detections = [d for d in detections if int(d[5]) == 0]
    return people_detections

def classify_sleep(person_crop):
    processed_crop = preprocess_image(person_crop, target_size=(224, 224), grayscale=False)
    sleep_pred = sleep_model.predict(processed_crop)
    return np.argmax(sleep_pred) == 1

def predict_age(person_crop):
    processed_crop = preprocess_image(person_crop, target_size=(128, 128), grayscale=True)
    age_pred = age_model.predict(processed_crop)
    return round(age_pred[0][0])

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    results = []

    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w]
        status = 'Awake'
        age = None

        if classify_sleep(face_crop):
            status = 'Sleeping'
            age = predict_age(face_crop)

        results.append({'bbox': [x, y, x+w, y+h], 'age': age, 'status': status})

    return results

def detect_age_and_sleep():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    frame = cv2.imread(file_path)
    results = process_frame(frame)
    display_results(frame, results)

def display_results(frame, results):
    message = ""
    for idx, result in enumerate(results):
        x1, y1, x2, y2 = result['bbox']
        label = f"Person-{idx + 1}, {result['status']}"
        if result['status'] == 'Sleeping':
            color = (0, 0, 255)  # Red
            label += f", Age: {result['age']}"
        else:
            color = (0, 255, 0)  # Green

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Person-{idx + 1}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, f"{result['status']}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if result['age'] is not None:
            cv2.putText(frame, f"Age: {result['age']}", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        message += f"Person-{idx + 1}, Status: {result['status']}, Age: {result['age']}\n"

    # Resize the frame to fit in the tkinter window
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb).resize((400, 300))  # Resize to fit in the GUI
    img_tk = ImageTk.PhotoImage(img_pil)

    # Update the tkinter image label
    label_img.config(image=img_tk)
    label_img.image = img_tk

    # Update the status message
    label_status.config(text=message)

# Creating the main window
root = tk.Tk()
root.geometry('800x600')
root.title("Age and Sleep Detection in Cars")
root.configure(bg="#b0e0e6")

# Creating a button for detecting age and sleep
button_detect = tk.Button(root, text="DETECT", command=detect_age_and_sleep, bg="#90ee90", width=15, height=2)
button_detect.place(x=50, y=200)

# Creating an exit button
button_exit = tk.Button(root, text="EXIT", command=root.quit, bg="#ff6961", width=15, height=2)
button_exit.place(x=50, y=260)

# Creating a label for displaying the uploaded image
label_img = tk.Label(root, bg="#b0e0e6")
label_img.place(x=300, y=120)

# Creating a label for displaying the status
label_status = tk.Label(root, text="None", font=("Arial", 12), bg="#b0e0e6")
label_status.place(x=300, y=420)

# Creating a title label
label_title = tk.Label(root, text="INPUT IMAGE", font=("Arial", 20), bg="#b0e0e6")
label_title.place(x=370, y=50)

# Run the application
root.mainloop()