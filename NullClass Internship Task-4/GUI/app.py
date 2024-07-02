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

# Loadingthe models
model_expression = load_model("Machine Learning/Animal Emotion Detection/Saved Model/facial_expression_model_v2.0.h5")
model_animal = load_model("Machine Learning/Animal Identification/Saved Model/Animal_Detection_Model_v4.0.h5")

df = pd.read_csv('GUI/animal_name.csv')
ANIMAL_LIST = df['animal_name'].tolist()

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            old_images = glob.glob(os.path.join(app.root_path, 'static', '*'))
            for old_image in old_images:
                os.remove(old_image)

            filename = secure_filename(image_file.filename)
            image_location = os.path.join(app.root_path, 'static', filename)
            image_file.save(image_location)

            animal_pred = DetectAnimal(image_location, model_animal)
            emotion_pred = DetectEmotion(image_location, model_expression)

            combined_prediction = f"{emotion_pred} {animal_pred}"

            return render_template('index.html', prediction=combined_prediction, image_loc=filename)
    return render_template('index.html', prediction=None, image_loc='default_image.jpg')

def DetectAnimal(file_path, model):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (224, 224))
    img_array = img.reshape((1, 224, 224, 3))

    predictions = model.predict(img_array)
    animal_label = ANIMAL_LIST[np.argmax(predictions[0])]

    return animal_label

def DetectEmotion(file_path, model):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img.reshape(1, 48, 48, 1)
    img = img / 255.0

    pred = model.predict(img)
    emotion_label = convert_to_emotion_label(pred[0])

    return emotion_label


def convert_to_emotion_label(prediction):
    labels = ['Happy', 'Sad', 'Hungry']
    return labels[np.argmax(prediction)]

@app.route("/home")
def test():
    return render_template('index2.html')


if __name__ == '__main__':
    app.run(debug=True, port=8000)
