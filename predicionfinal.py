# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 09:47:19 2024

@author: win10Davids
"""

import os
import tensorflow as tf
import librosa
from mp3towav import convert_to_wav

###########################################################################
#C:/Users/win10Davids/Documents/tesispanel/Dataset/wav_output_6_8
#just_path = "genres/blues/"
#song_path = "genres/blues/1.wav"
#song_name = "1"
just_path = "C:/Users/win10Davids/Documents/tesispanel/mp3/cuecasmp3/"
#song_path = "C:/Users/win10Davids/Documents/tesispanel/mp3/cuecasmp3/El Duraznero.mp3"
song_path = "C:/Users/win10Davids/Documents/tesispanel/mp3/notcuecamp3/95. Don Omar Ft. Aventura - Ella Y Yo (PORTAL DJ BOLIVIA).mp3"
song_name = "1"
##########################################################################

# Constants which depend on the model. If you train the model with different values,
# need to change those values here too
num_mfcc = 13
n_fft = 2048
hop_length = 512
sample_rate = 22050
samples_per_track = sample_rate * 30
num_segment = 10
############################################################################

if __name__ == "__main__":
    model = tf.keras.models.load_model("45cuecamodel_RNN_LSTM.h5")
    # model.summary()

    # classes = ["Blues","Classical","Country","Disco","Hiphop",
    #            "Jazz","Metal","Pop","Reggae","Rock"]
    classes = ["wav_notcueca", "wav_output_6_8"]

    class_predictions = []

    samples_per_segment = int(samples_per_track / num_segment)

    if song_path.endswith('.mp3'):
        path_to_save = just_path + song_name + ".wav"
        convert_to_wav(song_path, path_to_save)
        song_path = path_to_save
    else:
        pass

    # load the song
    x, sr = librosa.load(song_path, sr=sample_rate)
    song_length = int(librosa.get_duration(path=song_path))

    prediction_per_part = []
    parts = 1  # Inicialización por defecto
    flag = 0
    if song_length > 30:
        print("Song is greater than 30 seconds")
        samples_per_track_30 = sample_rate * song_length
        parts = int(song_length / 30)
        samples_per_segment_30 = int(samples_per_track_30 / parts)
        flag = 1
        print("Song sliced into " + str(parts) + " parts")
    elif song_length == 30:
        parts = 1
        flag = 0
    else:
        print("Too short, enter a song of length minimum 30 seconds")
        flag = 2

    if flag != 2:  # Proceed only if the song is not too short
        for i in range(parts):
            if flag == 1:
                print("Song snippet ", i + 1)
                start30 = samples_per_segment_30 * i
                finish30 = start30 + samples_per_segment_30
                y = x[start30:finish30]
                # print(len(y))
            elif flag == 0:
                print("Song is 30 seconds, no slicing")
                y = x

            for n in range(num_segment):
                start = samples_per_segment * n
                finish = start + samples_per_segment
                # print(len(y[start:finish]))
                mfcc = librosa.feature.mfcc(y=y[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                mfcc = mfcc.T
                # print(mfcc.shape)
                mfcc = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1])
                # print(mfcc.shape)
                array = model.predict(mfcc) * 100
                array = array.tolist()

                # find maximum percentage class predicted
                class_predictions.append(array[0].index(max(array[0])))

            occurence_dict = {}
            for i in class_predictions:
                if i not in occurence_dict:
                    occurence_dict[i] = 1
                else:
                    occurence_dict[i] += 1

            max_key = max(occurence_dict, key=occurence_dict.get)
            prediction_per_part.append(classes[max_key])

        # print(prediction_per_part)
        prediction = max(set(prediction_per_part), key=prediction_per_part.count)
        print(prediction)
    else:
        print("Song is too short to process")
