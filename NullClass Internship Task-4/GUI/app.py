import os
import glob
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd

app = Flask(__name__)

# Load the models directly from .h5 files
model_expression = load_model("Machine Learning/Animal Emotion Detection/Saved Model/facial_expression_model_v2.0.h5")
model_animal = load_model("Machine Learning/Animal Identification/Saved Model/Animal_Detection_Model_v4.0.h5")

# Load the CSV file for animal names
df = pd.read_csv('GUI/animal_name.csv')
ANIMAL_LIST = df['animal_name'].tolist()

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            # Delete the old image file
            old_images = glob.glob(os.path.join(app.root_path, 'static', '*'))
            for old_image in old_images:
                os.remove(old_image)

            # Save the new image file
            filename = secure_filename(image_file.filename)
            image_location = os.path.join(app.root_path, 'static', filename)
            image_file.save(image_location)

            # Detect animal and emotion
            animal_pred = DetectAnimal(image_location, model_animal)
            emotion_pred = DetectEmotion(image_location, model_expression)

            # Concatenate animal and emotion predictions
            combined_prediction = f"{emotion_pred} {animal_pred}"

            return render_template('index.html', prediction=combined_prediction, image_loc=filename)
    return render_template('index.html', prediction=None, image_loc='default_image.jpg')

def DetectAnimal(file_path, model):
    # Read the image in color
    img = cv2.imread(file_path)
    # Resize the image to 224x224 (the input size that your model expects)
    img = cv2.resize(img, (224, 224))
    # Reshape the image to add a batch dimension
    img_array = img.reshape((1, 224, 224, 3))

    # Predict
    predictions = model.predict(img_array)
    # Convert predictions to meaningful labels using the CSV file data
    animal_label = ANIMAL_LIST[np.argmax(predictions[0])]

    return animal_label

def DetectEmotion(file_path, model):
    # Read the image in grayscale
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    # Resize the image to 48x48 (the input size that your model expects)
    img = cv2.resize(img, (48, 48))
    # Reshape the image to match the input shape of the model: (1, 48, 48, 1)
    img = img.reshape(1, 48, 48, 1)
    # Normalize the image if your model expects normalized inputs
    img = img / 255.0

    # Predict
    pred = model.predict(img)
    # Convert predictions to meaningful labels
    emotion_label = convert_to_emotion_label(pred[0]) # Implement this function based on your labels

    return emotion_label


def convert_to_emotion_label(prediction):
    # Map your prediction indices to actual emotions
    labels = ['Happy', 'Sad', 'Hungry'] # Replace with your actual labels
    return labels[np.argmax(prediction)]

# The rest of your Flask app code remains the same

if __name__ == '__main__':
    app.run(debug=True, port=8000)
