import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import joblib
import numpy as np
import librosa

# Load the model and preprocessing objects
model_path = 'Machine Learning/Speech Gender Detection/Saved Model/gender_detectionwith_voice_model_v1.0.pkl'
scaler_path = 'Machine Learning/Speech Gender Detection/Saved polynomials and transformers/scaler_v1.0.pkl'
poly_path = 'Machine Learning/Speech Gender Detection/Saved polynomials and transformers/poly_v1.0.pkl'
selected_indices_path = 'Machine Learning/Speech Gender Detection/Saved polynomials and transformers/selected_indices_v1.0.pkl'

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
poly = joblib.load(poly_path)
selected_indices = joblib.load(selected_indices_path)

def preprocess_input(raw_features, scaler, poly, selected_indices):
    # Scale the raw features
    scaled_features = scaler.transform(np.array([raw_features]))
    
    # Apply polynomial transformation
    poly_features = poly.transform(scaled_features)
    
    # Use only the selected features (align with training phase)
    final_features = poly_features[:, selected_indices]
    
    return final_features

def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    features = [
        np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        np.std(librosa.feature.spectral_centroid(y=y, sr=sr)),
        np.median(librosa.feature.spectral_centroid(y=y, sr=sr)),
        np.percentile(librosa.feature.spectral_centroid(y=y, sr=sr), 25),
        np.percentile(librosa.feature.spectral_centroid(y=y, sr=sr), 75),
        np.percentile(librosa.feature.spectral_centroid(y=y, sr=sr), 75) - np.percentile(librosa.feature.spectral_centroid(y=y, sr=sr), 25),
        np.mean(librosa.feature.spectral_centroid(y=y, sr=sr) - np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))) ** 3,
        np.mean(librosa.feature.spectral_centroid(y=y, sr=sr) - np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))) ** 4,
        np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        np.mean(librosa.feature.spectral_flatness(y=y)),
        np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        np.mean(librosa.effects.harmonic(y)),
        np.min(librosa.effects.harmonic(y)),
        np.max(librosa.effects.harmonic(y)),
        np.mean(librosa.effects.percussive(y)),
        np.min(librosa.effects.percussive(y)),
        np.max(librosa.effects.percussive(y)),
        np.max(librosa.effects.percussive(y)) - np.min(librosa.effects.percussive(y)),
        np.mean(librosa.effects.harmonic(y)) / (np.mean(librosa.effects.percussive(y)) + 1e-6)
    ]
    return features

def predict_gender(audio_path):
    features = extract_features(audio_path)
    preprocessed_features = preprocess_input(features, scaler, poly, selected_indices)
    prediction = model.predict(preprocessed_features)
    return prediction

def upload_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        gender = predict_gender(file_path)
        messagebox.showinfo("Prediction Result", f'The predicted gender is: {gender[0]}')

# Tkinter window
root = tk.Tk()
root.title("Gender Detection from Audio")

upload_button = tk.Button(root, text="Upload Audio File", command=upload_file)
upload_button.pack(pady=20)

close_button = tk.Button(root, text="Close", command=root.quit)
close_button.pack(pady=20)

root.mainloop()
