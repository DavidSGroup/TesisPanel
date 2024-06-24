# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 01:49:13 2024

@author: win10Davids
"""

import os
import json
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import librosa

# Rutas a los datos
dataset_path = "Dataset"
#jsonpath = "jsonCuecas"
jsonpath = "cdata.json"

# Función para cargar datos desde el archivo JSON
def load_data(json_path):
    with open(json_path, "r") as fp:
        data = json.load(fp)
    x = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return x, y

# Función para preparar los datasets de entrenamiento, validación y prueba
def prepare_datasets(test_size, val_size, json_path):
    x, y = load_data(json_path)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size)
    return x_train, x_val, x_test, y_train, y_val, y_test

# Construir el modelo
def build_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(tf.keras.layers.LSTM(64))
    model.add(tf.keras.layers.Dropout(0.3))  # Añadir Dropout para evitar sobreajuste
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))  # Ajuste para dos clases
    return model

# Preparar los datasets
x_train, x_val, x_test, y_train, y_val, y_test = prepare_datasets(0.3, 0.2, jsonpath)

# Definir la forma de entrada
input_shape = (x_train.shape[1], x_train.shape[2])

# Construir el modelo
model = build_model(input_shape)

# Compilar el modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Reducir la tasa de aprendizaje
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Resumen del modelo
model.summary()

# Entrenar el modelo
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=45, batch_size=64)  # Aumentar el número de épocas y el tamaño del lote
  # evaluate model on test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
#    print('\nTest accuracy:', test_acc)
    # Imprimir la precisión en porcentaje
print('\nTest accuracy: {:.2f}%'.format(test_acc * 100))

model.save("45cuecamodel_RNN_LSTM.h5")
    #model.save("C:/Users/win10Davids/Documents/202002-Keshri/Source Code/lstm.h5")
print("Saved model to disk")
    