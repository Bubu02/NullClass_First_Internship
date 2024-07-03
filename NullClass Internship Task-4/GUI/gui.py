import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import cv2
import json

# Load paths from the JSON file
with open('PATHS/paths.json', 'r') as f:
    paths = json.load(f)

animal_model_path = paths['animal_model_path']
emotion_model_path = paths['emotion_model_path']
animal_labels_path = paths['animal_labels_path']

# Load the trained emotion model
emotion_model = load_model(emotion_model_path)

# Load animal labels from CSV
animal_labels_df = pd.read_csv(animal_labels_path)
animal_labels = animal_labels_df['animal_name'].tolist()

emotion_labels = ["Happy", "Sad", "Hungry"]

yolo_config_path = paths['yolo_config_path']
yolo_weights_path = paths['yolo_weights_path']
yolo_labels_path = paths['yolo_labels_path']

with open(yolo_labels_path, 'r') as f:
    yolo_labels = f.read().strip().split("\n")

net = cv2.dnn.readNetFromDarknet(yolo_config_path, yolo_weights_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def preprocess_image(image, target_size=(224, 224), grayscale=False):
    img = cv2.resize(image, target_size)
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape((img.shape[0], img.shape[1], 1))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def detect_animals(image):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, w, h) = box.astype("int")
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    animal_detections = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            if yolo_labels[class_ids[i]] in animal_labels:
                x, y, w, h = boxes[i]
                animal_detections.append({
                    'bbox': [x, y, x+w, y+h],
                    'label': animal_labels.index(yolo_labels[class_ids[i]])
                })
    return animal_detections

def classify_emotion(animal_crop):
    processed_crop = preprocess_image(animal_crop, target_size=(224, 224), grayscale=True)
    emotion_pred = emotion_model.predict(processed_crop)
    return np.argmax(emotion_pred)

def process_frame(frame):
    results = detect_animals(frame)
    processed_results = []

    for result in results:
        x1, y1, x2, y2 = result['bbox']
        animal_crop = frame[y1:y2, x1:x2]
        emotion = classify_emotion(animal_crop)
        animal_label = animal_labels[result['label']]
        processed_results.append({
            'bbox': [x1, y1, x2, y2],
            'emotion': emotion_labels[emotion],
            'animal_label': animal_label
        })

    return processed_results

def detect_animals_and_emotions():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
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
        emotion = result['emotion']
        animal_label = result['animal_label']

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Animal-{idx + 1}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Type: {animal_label}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Emotion: {emotion}", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        message += f"Animal-{idx + 1}, Type: {animal_label}, Emotion: {emotion}\n"

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb).resize((700, 500))  # Adjusted to fit in the GUI
    img_tk = ImageTk.PhotoImage(img_pil)

    label_img.config(image=img_tk)
    label_img.image = img_tk

    label_status.config(text=message)

root = tk.Tk()
root.title("Animal Emotion Detection")
root.configure(bg="#b0e0e6")
root.state('zoomed')

frame_left = tk.Frame(root, width=300, bg="#b0e0e6")
frame_left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

frame_original = tk.Frame(frame_left, width=300, height=300, bg="#d3d3d3")
frame_original.pack(pady=10)

label_original_img = tk.Label(frame_original, bg="#d3d3d3", text="PREVIEW IMAGE", font=("Arial", 16))
label_original_img.pack(expand=True)

frame_buttons = tk.Frame(frame_left, width=300, bg="white")
frame_buttons.pack(pady=10, fill=tk.X)

button_upload = tk.Button(frame_buttons, text="DETECT", command=detect_animals_and_emotions, bg="#90ee90", width=15, height=2)
button_upload.grid(row=0, column=0, padx=5, pady=5)

button_exit = tk.Button(frame_buttons, text="EXIT", command=root.quit, bg="#ff6961", width=15, height=2)
button_exit.grid(row=0, column=1, padx=5, pady=5)

label_status = tk.Label(frame_buttons, text="", font=("Arial", 12), bg="white", anchor='nw', justify='left')
label_status.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

frame_right = tk.Frame(root, bg="#d3d3d3")
frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

label_img = tk.Label(frame_right, bg="#d3d3d3", text="PROCESSED IMAGE", font=("Arial", 16))
label_img.pack(expand=True)

root.mainloop()
