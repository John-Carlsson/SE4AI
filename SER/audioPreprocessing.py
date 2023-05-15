import librosa
import os
import soundfile as sf
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


Crema = "AudioWAV"
crema_path = os.listdir(Crema)


def display_dataset():
    file_emotion = []
    file_path = []

    for file in crema_path:
        # storing file paths
        file_path.append(Crema + file)
        # storing file emotions
        part = file.split('_')
        if part[2] == 'SAD':
            file_emotion.append('sad')
        elif part[2] == 'ANG':
            file_emotion.append('angry')
        elif part[2] == 'DIS':
            file_emotion.append('disgust')
        elif part[2] == 'FEA':
            file_emotion.append('fear')
        elif part[2] == 'HAP':
            file_emotion.append('happy')
        elif part[2] == 'NEU':
            file_emotion.append('neutral')
        else:
            file_emotion.append('Unknown')
        # pre-processing/data-augmentation

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    print(f"There are {len(file_path)} audios in the dataset.")
    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    # Crema_df = pd.concat([emotion_df, path_df], axis=1)
    # Crema_df.head()
    # display(Crema_df)

    return path_df, emotion_df
