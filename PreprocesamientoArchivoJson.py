# -*- coding: utf-8 -*-
##funciona genera el arcivo json
"""
Created on Fri Jun 21 23:38:43 2024

@author: win10Davids
"""

import librosa
import os
import math
import json
#dataset_path = "C:/Users/win10Davids/Documents/202002-Keshri/dataset/Data/genres_original"
#jsonpath = "C:/Users/win10Davids/Documents/202002-Keshri/Source Code/1data_json"

dataset_path = "Dataset"
jsonpath = "jsonCuecas"

sample_rate = 22050
samples_per_track = sample_rate * 30

def preprocess(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segment=5):
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(samples_per_track / num_segment)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath != dataset_path:
            # Adding all the labels
            label = os.path.basename(dirpath)
            if label not in data["mapping"]:
                data["mapping"].append(label)
            label_index = data["mapping"].index(label)

            # Going through each song within a label
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                y, sr = librosa.load(file_path, sr=sample_rate)

                # Cutting each song into segments
                for n in range(num_segment):
                    start = samples_per_segment * n
                    finish = start + samples_per_segment
                    mfcc = librosa.feature.mfcc(y=y[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T  # 259 x 13

                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(label_index)
                        print("Processed Track: ", file_path, "Segment:", n + 1)

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
    print("Data has been successfully saved to", json_path)

if __name__ == "__main__":
    preprocess(dataset_path, jsonpath, num_segment=10)
