import os
import glob
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import model_from_json

app = Flask(__name__)

def FacialExpressionModel(json_file, weights_file):
    with open(json_file,"r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
 
    model.load_weights(weights_file)
    model.compile(optimizer ='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

    return model

# Load the model
model = FacialExpressionModel("Saved Model\Trained_model_kaggle.json","Model Weights\kaggle_model.weights.h5")

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

            pred = Detect(image_location)
            return render_template('index.html', prediction=pred, image_loc=filename)
    return render_template('index.html', prediction=None, image_loc='default_image.jpg')

def Detect(file_path):
    facec = cv2.CascadeClassifier('.env/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    REGIONS_LIST = ["Region-0","Region-1","Region-2","Region-3","Region-4"]

    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image,1.3,5)

    if len(faces) == 0:
        return "Unable to detect a face in the image."

    for (x,y,w,h) in faces:
        fc = gray_image[y:y+h,x:x+w]
        roi = cv2.resize(fc,(48,48))
        pred = REGIONS_LIST[np.argmax(model.predict(roi[np.newaxis,:,:,np.newaxis]))]

    return pred

if __name__ == '__main__':
    app.run(debug=True, port=8000)
