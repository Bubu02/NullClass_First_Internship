import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import load_model

# Load the saved model
loaded_model = load_model("NullClass Internship Task-4/Machine Learning/Animal Emotion Detection/Saved Model/facial_expression_model_v2.0.h5")

# Function to load and preprocess image
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

# Function to predict emotion
def predict_emotion(img):
    prediction = loaded_model.predict(img)
    emotion_label = np.argmax(prediction[0])
    emotion_classes = {0: 'happy', 1: 'sad', 2: 'hungry'}
    predicted_emotion = emotion_classes[emotion_label]
    return predicted_emotion

# Function to open file dialog and load image
def upload_image():
    file_path = filedialog.askopenfilename()
    uploaded_image = Image.open(file_path)
    uploaded_image = uploaded_image.resize((250, 250), Image.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=uploaded_image)
    label_img.config(image=imgtk)
    label_img.image = imgtk

    # Predict emotion
    img = load_and_preprocess_image(file_path)
    emotion = predict_emotion(img)
    label_emotion.config(text=f"Predicted Emotion: {emotion}")

# Create main window
root = tk.Tk()

# Set window size
root.geometry('250x330')

# Create a button for uploading images
button_upload = tk.Button(root, text="Upload Image", command=upload_image)
button_upload.pack()

# Create a label for displaying the uploaded image
label_img = tk.Label(root)
label_img.pack()

# Create a label for displaying the predicted emotion
label_emotion = tk.Label(root, text="Predicted Emotion: ")
label_emotion.pack()

# Run the application
root.mainloop()
