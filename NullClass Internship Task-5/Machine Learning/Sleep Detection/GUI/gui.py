import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import numpy as np
import cv2

# Loading the trained model for detecting drowsiness
model = load_model("Machine Learning/Sleep Detection/Saved Models/Car_Drowsiness_Detection_Model_v2.0.h5")

def detect_drowsiness(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)  # Image in RGB format
    image = cv2.resize(image, (224, 224))  # Resizing the image to match your model's input size
    image = image.reshape((1, 224, 224, 3))  # Adding a batch dimension and three channels (RGB)
    pred = model.predict(image)
    status = "Sleeping" if np.argmax(pred) == 1 else "Awake"
    print("Predicted Status: " + status)
    label_status.config(text=f"Predicted Status: {status}")

def upload_image():
    file_path = filedialog.askopenfilename()
    uploaded_image = Image.open(file_path)
    uploaded_image = uploaded_image.resize((250, 250), Image.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=uploaded_image)
    label_img.config(image=imgtk)
    label_img.image = imgtk
    detect_drowsiness(file_path)

# Creating the main window
root = tk.Tk()
root.geometry('250x330')

# Creating a button for uploading images
button_upload = tk.Button(root, text="Upload Image", command=upload_image)
button_upload.pack()

# Creating a label for displaying the uploaded image
label_img = tk.Label(root)
label_img.pack()

# Creating a label for displaying the predicted status (Sleeping or Awake)
label_status = tk.Label(root, text="Predicted Status: ")
label_status.pack()

# Run the application
root.mainloop()
