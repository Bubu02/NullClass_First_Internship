import tkinter as tk
from tkinter import filedialog
# from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import numpy as np
import cv2
import pandas as pd

# Load the CSV file
df = pd.read_csv('NullClass Internship Task-4/Machine Learning/Animal Identification\GUI/animal_name.csv')

# def AnimalIdentificationModel(json_file, weights_file):
#     with open(json_file,"r") as file:
#         loaded_model_json = file.read()
#         model = model_from_json(loaded_model_json)
 
#     model.load_weights(weights_file)
#     model.compile(optimizer ='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

#     return model

# Load the model
model = load_model("NullClass Internship Task-4/Machine Learning/Animal Identification/Saved Model/Animal_Detection_Model_v4.0.h5")

# Get the list of animal names
ANIMAL_LIST = df['animal_name'].tolist()

def Detect(file_path):
    image = cv2.imread(file_path)  # Read the image in color
    image = cv2.resize(image, (224, 224))  # resize the image to 224x224
    image = image.reshape((1, 224, 224, 3))  # add a batch dimension
    pred = ANIMAL_LIST[np.argmax(model.predict(image))]
    print("Predicted Animal is " + pred)
    label_emotion.config(text=f"Predicted Animal: {pred}")


def upload_image():
    file_path = filedialog.askopenfilename()
    uploaded_image = Image.open(file_path)
    uploaded_image = uploaded_image.resize((250, 250), Image.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=uploaded_image)
    label_img.config(image=imgtk)
    label_img.image = imgtk
    Detect(file_path)
 
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

# Create a label for displaying the predicted animal
label_emotion = tk.Label(root, text="Predicted Animal: ")
label_emotion.pack()

# Run the application
root.mainloop()
