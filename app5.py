# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 21:32:16 2024

@author: win10Davids
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox
import tensorflow as tf
import librosa
from pydub import AudioSegment
from mp3towav import convert_to_wav

# Ruta al modelo guardado
MODEL_PATH = os.path.join(os.getcwd(), "model", "45cuecamodel_RNN_LSTM.h5")

# Cargar el modelo de TensorFlow
model = tf.keras.models.load_model(MODEL_PATH)

# Constantes del modelo
num_mfcc = 13
n_fft = 2048
hop_length = 512
sample_rate = 22050
samples_per_track = sample_rate * 30
num_segment = 10
classes = ["wav_notcueca", "wav_output_6_8"]

# Función para convertir MP3 a WAV
def convert_to_wav(src, dst):
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")

# Función para realizar la predicción de audio
def predict_audio(file_path, text_widget):
    # Convertir MP3 a WAV si es necesario
    if file_path.endswith('.mp3'):
        wav_path = file_path.replace('.mp3', '.wav')
        convert_to_wav(file_path, wav_path)
        file_path = wav_path

    # Cargar y procesar el archivo de audio
    x, sr = librosa.load(file_path, sr=sample_rate)
    song_length = int(librosa.get_duration(filename=file_path))
    samples_per_segment = int(samples_per_track / num_segment)
    prediction_per_part = []
    parts = 1

    if song_length > 30:
        samples_per_track_30 = sample_rate * song_length
        parts = int(song_length / 30)
        samples_per_segment_30 = int(samples_per_track_30 / parts)
        flag = 1
    elif song_length == 30:
        flag = 0
    else:
        text_widget.insert(tk.END, "Song is too short to process\n")
        return "Too short, enter a song of length minimum 30 seconds"

    class_predictions = []

    if flag != 2:  # Proceed only if the song is not too short
        for i in range(parts):
            if flag == 1:
                start30 = samples_per_segment_30 * i
                finish30 = start30 + samples_per_segment_30
                y = x[start30:finish30]
            elif flag == 0:
                y = x

            for n in range(num_segment):
                start = samples_per_segment * n
                finish = start + samples_per_segment
                mfcc = librosa.feature.mfcc(y=y[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                mfcc = mfcc.T
                mfcc = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1])
                array = model.predict(mfcc) * 100
                array = array.tolist()
                class_predictions.append(array[0].index(max(array[0])))
                text_widget.insert(tk.END, f"Processed segment {n+1}/{num_segment}\n")
                text_widget.see(tk.END)
                text_widget.update_idletasks()

            occurence_dict = {}
            for i in class_predictions:
                if i not in occurence_dict:
                    occurence_dict[i] = 1
                else:
                    occurence_dict[i] += 1

            max_key = max(occurence_dict, key=occurence_dict.get)
            prediction_per_part.append(classes[max_key])

        prediction = max(set(prediction_per_part), key=prediction_per_part.count)
        return prediction
    else:
        return "Song is too short to process"

# Función para seleccionar un archivo
def select_file():
    # Limpiar etiquetas y widget de texto antes de cargar un nuevo archivo
    file_label.config(text="No Hay Archivo seleccionado")
    result_label.config(text="El resultado de la predicción se mostrará aquí")
    text_widget.delete(1.0, tk.END)
    
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    if file_path:
        file_label.config(text=f"Seleccion de Archivo: {os.path.basename(file_path)}")
        text_widget.delete(1.0, tk.END)
        result = predict_audio(file_path, text_widget)
        
        if result == "wav_notcueca":
            message = f"El archivo {os.path.basename(file_path)} NO es Cueca"
        elif result == "wav_output_6_8":
            message = f"El archivo {os.path.basename(file_path)} SI es Cueca"
        else:
            message = result
        
        result_label.config(text=message)

# Configurar la ventana principal de Tkinter
root = tk.Tk()
root.title("Audio Classification")
root.geometry("600x400")

# Crear etiquetas para mostrar el archivo seleccionado y la predicción
file_label = tk.Label(root, text="No Hay Archivo seleccionado", wraplength=400)
file_label.pack(pady=10)

result_label = tk.Label(root, text="El resultado de la predicción se mostrará aquí", wraplength=400)
result_label.pack(pady=10)

# Crear una widget de texto para mostrar el progreso
text_widget = tk.Text(root, height=10, width=70)
text_widget.pack(pady=10)

# Crear un botón para cargar el archivo
btn_load = tk.Button(root, text="Cargar archivo de audio", command=select_file)
btn_load.pack(pady=20)

# Ejecutar la aplicación
root.mainloop()
