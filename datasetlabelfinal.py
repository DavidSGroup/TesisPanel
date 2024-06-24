# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 22:40:58 2024

@author: win10Davids
"""

import os
import librosa
import numpy as np
import pandas as pd

# Lista de carpetas a procesar
folders = [
    "C:/Users/win10Davids/Documents/tesispanel/Dataset/wav_output_6_8",
    "C:/Users/win10Davids/Documents/tesispanel/Dataset/wav_notcueca",
    # Añade más carpetas aquí si es necesario
]

# Lista para almacenar las características de todos los archivos
all_features = []

# Lista para almacenar archivos problemáticos
problematic_files = []

# Iterar sobre todas las carpetas
for folder_path in folders:
    folder_name = os.path.basename(folder_path)  # Obtener el nombre de la carpeta
    # Iterar sobre todos los archivos en la carpeta
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            try:
                y, sr = librosa.load(file_path, sr=None)
                
                # Verificar que el archivo no esté vacío
                if len(y) == 0:
                    print(f"Error: El archivo {filename} está vacío o contiene solo silencio.")
                    problematic_files.append(file_path)
                    continue
                
                # Extraer características
                length = len(y)
                chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                rms = librosa.feature.rms(y=y)
                spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
                harmony = librosa.effects.harmonic(y)
                perceptr = librosa.effects.percussive(y)
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

                # Calcular las medias y varianzas
                features = {
                    'filename': filename,
                    'length': length,
                    'chroma_stft_mean': np.mean(chroma_stft),
                    'chroma_stft_var': np.var(chroma_stft),
                    'rms_mean': np.mean(rms),
                    'rms_var': np.var(rms),
                    'spectral_centroid_mean': np.mean(spectral_centroid),
                    'spectral_centroid_var': np.var(spectral_centroid),
                    'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
                    'spectral_bandwidth_var': np.var(spectral_bandwidth),
                    'rolloff_mean': np.mean(rolloff),
                    'rolloff_var': np.var(rolloff),
                    'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
                    'zero_crossing_rate_var': np.var(zero_crossing_rate),
                    'harmony_mean': np.mean(harmony),
                    'harmony_var': np.var(harmony),
                    'perceptr_mean': np.mean(perceptr),
                    'perceptr_var': np.var(perceptr),
                    'tempo': tempo,
                }

                for i in range(1, 21):
                    features[f'mfcc{i}_mean'] = np.mean(mfcc[i-1])
                    features[f'mfcc{i}_var'] = np.var(mfcc[i-1])

                # Añadir la etiqueta 'label' al final
                features['label'] = folder_name

                # Añadir las características del archivo actual a la lista
                all_features.append(features)
                print(f"OK: Procesado el archivo {filename} en la carpeta {folder_name}")
            except Exception as e:
                print(f"Error procesando el archivo {filename} en la carpeta {folder_name}: {e}")
                problematic_files.append(file_path)

# Convertir a DataFrame y guardar directamente en un archivo CSV
output_path = "C:/Users/win10Davids/Documents/tesispanel/features_30.csv"
df = pd.DataFrame(all_features)
df.to_csv(output_path, index=False, sep=',')

# Mostrar el DataFrame
#·import ace_tools as tools; tools.display_dataframe_to_user(name="Características de Audio", dataframe=df)

# Mostrar archivos problemáticos
if problematic_files:
    print("Los siguientes archivos tuvieron problemas y no se procesaron correctamente:")
    for file in problematic_files:
        print(file)
