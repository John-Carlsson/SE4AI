import librosa
import os
import soundfile as sf
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


Crema = "AudioWAV"
crema_path = os.listdir(Crema)


stretched_audios_paths = "stretched_audios"
pitched_audios_paths = "pitched_audios"
padded_audios_paths = "padded_audios"

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


def load_audio(file_path):
    y, sr = librosa.load(file_path)
    return y, sr


def stretching_time():

    for file_name in crema_path:
        file_path = os.path.join(Crema, file_name)
        y, sr = load_audio(file_path)
        stretched_audio = librosa.effects.time_stretch(y=y, rate=random.randint(1, 3))
        new_file_name = os.path.basename(file_path)
        out_file_name = os.path.splitext(new_file_name)[0] + '_stretched.wav'
        out_path = os.path.join(stretched_audios_paths, out_file_name)
        if os.path.exists(out_path):
            os.remove(out_path)
        sf.write(out_path, stretched_audio, sr)

def pitching_voice():

    for file_name in crema_path:
        file_path = os.path.join(Crema, file_name)

        y, sr = load_audio(file_path)
        pitched_audio = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=random.randint(1, 6))
        new_file_name = os.path.basename(file_path)
        out_file_name = os.path.splitext(new_file_name)[0] + '_pitched.wav'
        out_path = os.path.join(pitched_audios_paths, out_file_name)
        if os.path.exists(out_path):
            os.remove(out_path)
        sf.write(out_path, pitched_audio, sr)

def adding_padding():


    target_duration = 6.0

    for file_name in crema_path:
        file_path = os.path.join(Crema, file_name)
        y, sr = load_audio(file_path)

        #convert the desired duration from seconds to samples and then we know how many samples are missing
        target_length = int(target_duration * sr)

        current_length = len(y)
        padding_length = max(0, target_length - current_length)
        padding = np.zeros(padding_length)

        padded_audio = np.concatenate((y, padding))

        new_file_name = os.path.basename(file_path)
        out_file_name = os.path.splitext(new_file_name)[0] + '_padded.wav'
        out_path = os.path.join(padded_audios_paths, out_file_name)
        if os.path.exists(out_path):
            os.remove(out_path)
        sf.write(out_path, padded_audio, sr)


def create_mel_spectrogram():
    mel_spectrograms = []

    for file_name in crema_path:
        file_path = os.path.join(Crema, file_name)
        y, sr = load_audio(file_path)

        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # define x-axis (time) and y-axis (frequency) values
        frames = range(mel_spectrogram_db.shape[1])
        t = librosa.frames_to_time(frames)
        f = librosa.mel_frequencies(n_mels=mel_spectrogram_db.shape[0], fmin=0, fmax=sr / 2)

        mel_spectrograms.append(mel_spectrogram_db)


        #display of the spectorgram if necessary
        #librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel', x_coords=t, y_coords=f)
        #plt.colorbar(format='%+2.0f dB')
        #plt.title('Mel Spectrogram')
        #plt.show()


    #irgendwie noch probleme mit dem convert to np array
    #mel_spectrograms_array = np.array(mel_spectrograms)



if __name__ == "__main__":
    display_dataset()
    stretching_time()