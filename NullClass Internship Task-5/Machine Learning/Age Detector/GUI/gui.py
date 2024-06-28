import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from keras.losses import MeanAbsoluteError
from PIL import Image, ImageTk
import numpy as np

# Loading the trained model for age detection
custom_objects = {
    'mae': MeanAbsoluteError()
}

model = load_model("Machine Learning/Age Detector/Saved Models/Age_Detection_Model_v1.1.h5", custom_objects=custom_objects)

def detect_age(file_path):
    img = Image.open(file_path)
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((128, 128), Image.LANCZOS)
    img = np.array(img, dtype=float)
    img /= 255.0
    img = img.reshape((1, 128, 128, 1))  # Add batch dimension and channel dimension
    pred = model.predict(img)
    age = pred[0][0]  # Directly use the predicted value for regression
    rounded_age = round(age)  # Round the age to the nearest integer
    print("Predicted Age: " + str(rounded_age))
    label_status.config(text=f"Predicted Age: {rounded_age}")

def upload_image():
    file_path = filedialog.askopenfilename()
    uploaded_image = Image.open(file_path)
    uploaded_image = uploaded_image.resize((250, 250), Image.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=uploaded_image)
    label_img.config(image=imgtk)
    label_img.image = imgtk
    detect_age(file_path)

# Creating the main window
root = tk.Tk()
root.geometry('250x330')

# Creating a button for uploading images
button_upload = tk.Button(root, text="Upload Image", command=upload_image)
button_upload.pack()

# Creating a label for displaying the uploaded image
label_img = tk.Label(root)
label_img.pack()

# Creating a label for displaying the predicted age
label_status = tk.Label(root, text="Predicted Age: ")
label_status.pack()

# Run the application
root.mainloop()
