import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import numpy as np
import cv2
import threading
import logging
from keras.losses import MeanAbsoluteError

logging.basicConfig(level=logging.INFO)

custom_objects = {
    'mae': MeanAbsoluteError()
}

try:
    sleep_model = load_model('Machine Learning/Sleep Detection/Saved Models/Car_Drowsiness_Detection_Model_v2.0.h5')
    age_model = load_model('Machine Learning/Age Detector/Saved Models/Age_Detection_Model_v1.1.h5', custom_objects=custom_objects)
    logging.info("Models loaded successfully.")
except Exception as e:
    logging.error(f"Error loading models: {e}")

# Haar Cascade for face detection
haarcascade_path = 'Main GUI/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haarcascade_path)

def preprocess_image(image, target_size=(224, 224), grayscale=False):
    img = cv2.resize(image, target_size)
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape((img.shape[0], img.shape[1], 1))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

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
    threading.Thread(target=process_and_display, args=(file_path,)).start()

def process_and_display(file_path):
    frame = cv2.imread(file_path)
    results = process_frame(frame)
    display_results(frame, results)
    display_original_image(file_path)

def display_original_image(file_path):
    img = Image.open(file_path).resize((300, 300))  # Adjusted to fit the frame
    img_tk = ImageTk.PhotoImage(img)
    label_original_img.config(image=img_tk)
    label_original_img.image = img_tk

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
        cv2.putText(frame, f"Person-{idx + 1}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"{result['status']}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if result['age'] is not None:
            cv2.putText(frame, f"Age: {result['age']}", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        message += f"Person-{idx + 1}, Status: {result['status']}, Age: {result['age']}\n"

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb).resize((700, 500))
    img_tk = ImageTk.PhotoImage(img_pil)

    label_img.config(image=img_tk)
    label_img.image = img_tk

    label_status.config(text=message)

root = tk.Tk()
root.title("Age and Sleep Detection in Cars")
root.configure(bg="#b0e0e6")
root.state('zoomed')

frame_left = tk.Frame(root, width=300, bg="#b0e0e6")
frame_left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

frame_original = tk.Frame(frame_left, width=300, height=300, bg="#d3d3d3")
frame_original.pack(pady=10)

# Creating a label for the original image preview
label_original_img = tk.Label(frame_original, bg="#d3d3d3", text="PREVIEW IMAGE", font=("Arial", 16))
label_original_img.pack(expand=True)

frame_buttons = tk.Frame(frame_left, width=300, bg="white")
frame_buttons.pack(pady=10, fill=tk.X)

button_upload = tk.Button(frame_buttons, text="DETECT", command=detect_age_and_sleep, bg="#90ee90", width=15, height=2)
button_upload.grid(row=0, column=0, padx=5, pady=5)

# Creating an exit button
button_exit = tk.Button(frame_buttons, text="EXIT", command=root.quit, bg="#ff6961", width=15, height=2)
button_exit.grid(row=0, column=1, padx=5, pady=5)

label_status = tk.Label(frame_buttons, text="", font=("Arial", 12), bg="white", anchor='nw', justify='left')
label_status.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

frame_right = tk.Frame(root, bg="#d3d3d3")
frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

label_img = tk.Label(frame_right, bg="#d3d3d3", text="PROCESSED IMAGE", font=("Arial", 16))
label_img.pack(expand=True)

root.mainloop()
