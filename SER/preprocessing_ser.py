# imports 
import os
import librosa
import pandas as pd
import numpy as np

# Default paths
default_path_raw_data = "AudioWAV"
default_path_store = os.path.join("Preprocessed_data", "Dataframe_with_Spectrograms")

# Default map
emotion_hot_encode = {
    'ANG': (1., 0., 0., 0., 0., 0.),
    'DIS': (0., 1., 0., 0., 0., 0.),
    'FEA': (0., 0., 1., 0., 0., 0.),
    'HAP': (0., 0., 0., 1., 0., 0.),
    'NEU': (0., 0., 0., 0., 1., 0.),
    'SAD': (0., 0., 0., 0., 0., 1.)
}

## Final Dataframe Columns: "Data Sample", "Name", "Emotion Class", "Emotion Vector", "Padded Sample", "Spectrogram"



################################
# Methods for training a model #
################################


def load_data(path=default_path_raw_data, sample_rate=None, vector_mapping=emotion_hot_encode):
    """
    Loads the data from a directory. Extracts data samples, their name, their emotion class, the emotion vector, the highest number of frames possessed by at least one sample of the dataset.

    Returns
    -------
    loaded_data : pandas dataframe
        consisting of data samples, their name, their emotion class
    max_n_frames : int
        the highest number of frames possessed by at least one sample of the dataset; can be used as the target size in padding
        
    """
    samples = []
    names = []
    emotion_classes = []
    emotion_vectors = []

    f_rates = []
    n_frames = []

    for wav_file in os.listdir(path):
        audio_data, sr = librosa.load(os.path.join(path, wav_file), sr=sample_rate)
        emotion_class = wav_file.split("_")[2]
        
        samples.append(audio_data)
        names.append(wav_file)
        emotion_classes.append(emotion_class)
        emotion_vectors.append(vector_mapping[emotion_class])

        f_rates.append(sr)
        n_frames.append(len(audio_data))

    
    zipped = list(zip(samples, names, emotion_classes, emotion_vectors))
    loaded_data = pd.DataFrame(zipped, columns=['Data Sample', 'Name', 'Emotion Class', 'Emotion Vector'])

    max_n_frames = max(n_frames)
    return loaded_data, max_n_frames



def pad_with_zeros(data: pd.DataFrame, target_length=None):
    padded = []
    for sample in data["Data Sample"]:
        padded.append(librosa.util.pad_center(sample, size=target_length))

    padded_data = pd.concat([data, pd.DataFrame(list(zip(padded)), columns=['Padded Sample'])], axis=1)
    
    return padded_data



def data_augmentation(data=None):
    pass


def calculate_spectrograms(data: pd.DataFrame, padded=True, n_fft=1024, hop_length=512):
    data_spectrograms = []

    if padded:
        column = "Padded Sample"
    else:
        column = "Data Sample"

    for sample in data[column]:
        # Compute STFT magnitude
        stft = librosa.stft(sample, n_fft=n_fft, hop_length=hop_length) # gives us a 2D numpy array of the shape (n_freq_bins, n_time_frames)
        stft_mag, stft_phase = librosa.magphase(stft)  # extracts the magnitudes and the phases and returns as separate matrices; stft_phase is ignored at the moment
        # Convert to dB scale
        stft_mag_db = librosa.amplitude_to_db(stft_mag, ref=np.max)
        # Turn to absolute values
        mag_spec = np.abs(stft_mag_db) 
        # Reshape to 3D, where the third dimension is number of channels (here: 1)
        reshaped = np.reshape(mag_spec, (*mag_spec.shape, 1))

        data_spectrograms.append(reshaped)

    spectrogram_shape = (mag_spec.shape[0], mag_spec.shape[1], 1)
    print("The shape of the spectrograms: ", spectrogram_shape)
     
    data_spectrograms = pd.concat([data, pd.DataFrame(list(zip(data_spectrograms)), columns=['Spectrogram'])], axis=1)
    
    return data_spectrograms, spectrogram_shape
    



def store_preprocessed_data(data: pd.DataFrame, path=default_path_store):
    data.to_pickle(path)


def load_spectrograms(path=default_path_store):
    """
    Loads the spectrograms from a pickle file. Extracts the shape of the first spectrogram in the dataframe.

    Returns
    -------
    spec_def : pandas dataframe
        columns = "Data Sample", "Name", "Emotion Class", "Emotion Vector", "Padded Sample", "Spectrogram"
    spec_shape : tuple
        the three dimensional shape of the first spectrogram in the dataframe: (number of frequency bins, number of time frames, number of channels)
        
    """
    spec_df = pd.read_pickle(path)
    spec_shape = (spec_df["Spectrogram"].iloc[0].shape[0], spec_df["Spectrogram"].iloc[0].shape[1], 1)
    return spec_df, spec_shape





###############################################
# Methods for using a model on real-life data #
###############################################


# Requirement: all input samples have the same duration and sample/frame rate
# Returns spectrograms
def preprocess():
    pass
    # Resampling
    # Normalization
    # Standardization
    # Calculating spectrograms













##########################
# Precofigured pipelines #
##########################

def load_pad_spec_store(path=default_path_raw_data):
    data, max_n_frames = load_data(path)
    padded_data = pad_with_zeros(data, max_n_frames)
    #data_augmentation()
    spec_df, spec_shape = calculate_spectrograms(padded_data)
    store_preprocessed_data(spec_df)








if __name__ == "__main__":
    load_pad_spec_store()
    specs = load_spectrograms()
    print(specs)
