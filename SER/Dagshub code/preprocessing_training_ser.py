import os
import librosa
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


### Default paths ###
# Raw data
default_path_raw_cremad = os.path.join(os.pardir, "data", "cremad")
default_path_raw_emodb = os.path.join(os.pardir, "data", "emodb")
# Preprocessed data
default_path_preprocessed_cremad = os.path.join(os.pardir, "data", "preprocessed_cremad.pickle")
default_path_preprocessed_emodb = os.path.join(os.pardir, "data", "preprocessed_emodb.pickle")


### Default emotion mapping ###
emotion_hot_encode_cremad = {
    'ANG': (1., 0., 0., 0., 0., 0.),
    'DIS': (0., 1., 0., 0., 0., 0.),
    'FEA': (0., 0., 1., 0., 0., 0.),
    'HAP': (0., 0., 0., 1., 0., 0.),
    'NEU': (0., 0., 0., 0., 1., 0.),
    'SAD': (0., 0., 0., 0., 0., 1.)
}

emotion_hot_encode_emodb = {
    'W': (1., 0., 0., 0., 0., 0.),
    'E': (0., 1., 0., 0., 0., 0.),
    'A': (0., 0., 1., 0., 0., 0.),
    'F': (0., 0., 0., 1., 0., 0.),
    'L': (0., 0., 0., 0., 1., 0.),  # boredom & neutral are taken as identical
    'T': (0., 0., 0., 0., 0., 1.),
    'N': (0., 0., 0., 0., 1., 0.),
}


## Final Dataframe Columns: "Spectrogram", "Emotion Vector"



############################################
# Methods to apply before training a model #
############################################

def load_data(path: str, dataset_name="cremad", sample_rate=None, return_n_frames="mean", plot_lengths=False):
    """
    Loads the data from a directory. Extracts data samples and their emotion class, maps the emotion vector and calculates the highest number of frames possessed by at least one sample of the dataset.

    Returns
    -------
    loaded_data : pandas dataframe
        consisting of data samples, their mapped emotion vector
    max_n_frames : int
        the highest number of frames possessed by at least one sample of the dataset; can be used as the target size in padding
        
    """
    samples = []
    emotion_vectors = []
    n_frames = []

    for wav_file in os.listdir(path):
        audio_data, sr = librosa.load(os.path.join(path, wav_file), sr=sample_rate)
        if dataset_name == "cremad":
            emotion_class = wav_file.split("_")[2]
            vector_mapping = emotion_hot_encode_cremad
        elif dataset_name == "emodb":
            emotion_class = wav_file[5]
            vector_mapping = emotion_hot_encode_emodb
        else: 
            raise ValueError('{} does not exist.'.format(dataset_name))
        

        samples.append(audio_data)
        emotion_vectors.append(vector_mapping[emotion_class])

        n_frames.append(len(audio_data))

    zipped = list(zip(samples, emotion_vectors))
    loaded_data = pd.DataFrame(zipped, columns=['Data Sample', 'Emotion Vector'])

    print("Loaded %i data samples."%(loaded_data.shape[0]))
    if plot_lengths:
        plot_length_distribution(n_frames)

    if return_n_frames =="max": 
        max_n_frames = max(n_frames)
        return loaded_data, max_n_frames
    elif return_n_frames == "mean":
        mean_n_frames = np.mean(n_frames)
        return loaded_data, mean_n_frames
    else: 
        return loaded_data



def plot_length_distribution(lengths):
    """
    Plots the distribution of audio sample lengths.

    Parameters
    ----------
    lengths : list or array-like
        List or array containing the lengths of audio samples.
    """
    plt.hist(lengths, bins=20)  

    plt.xlabel('Length')
    plt.ylabel('Count')
    plt.title('Distribution of Audio Sample Lengths')
    plt.show()



def fix_length(data:pd.DataFrame, target_length):
    """
    Fixes the length of data samples in a DataFrame to a specified target length using librosa.util.fix_length.

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame containing the data samples and their associated emotion vectors.
    target_length : int
        The desired target length for the data samples after fixing.

    Returns
    -------
    fixed_data : pandas DataFrame
        DataFrame with the fixed-length data samples and their associated emotion vectors.

    Notes
    -----
    - This function uses librosa.util.fix_length to fix the length of each data sample in the "Data Sample" column.
    - The fixed-length data samples are stored in the "Data Sample" column of the returned DataFrame.
    - The associated emotion vectors from the input DataFrame are preserved in the returned DataFrame.

    """
    fixed = []
    for sample in data["Data Sample"]:
        fixed.append(librosa.util.fix_length(sample, size=int(target_length)))
    
    fixed = pd.DataFrame(list(zip(fixed)), columns=["Data Sample"])
    fixed_data = pd.concat([fixed, data["Emotion Vector"]], axis=1)

    return fixed_data


def add_noise(data: pd.DataFrame, noise_level=0.5):
    """
    Adds noise to the data samples in a pandas DataFrame.

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame containing the data samples and their associated emotion vectors.
    noise_level : float, optional
        The noise level to apply to the data samples.
        A value between 0 and 1, where 0 means no noise and 1 means full noise.
        Defaults to 0.5.

    Returns
    -------
    noisy_data : pandas DataFrame
        DataFrame with the noisy data samples and their associated emotion vectors.

    Notes
    -----
    - This function adds random noise to each data sample in the "Data Sample" column of the input DataFrame.
    - The noisy data samples are stored in the "Data Sample" column of the returned DataFrame.
    - The associated emotion vectors from the input DataFrame are preserved in the returned DataFrame.

    """
    noisy_data = []
    for sample in data["Data Sample"]:
        noise = np.random.uniform(-1, 1, len(sample))
        noise *= noise_level
        noisy_data.append(sample + noise)
    
    noisy_data = pd.DataFrame(list(zip(noisy_data)), columns=["Data Sample"])
    noisy_data = pd.concat([noisy_data, data["Emotion Vector"]], axis=1)
    
    complete = pd.concat([data, noisy_data], ignore_index=True)
    print(complete)


    return complete



def calculate_spectrograms(data: pd.DataFrame, n_fft=1024, hop_length=512, return_spec_shape=True):
    """
    Calculates spectrograms from the data samples in a DataFrame using librosa.

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame containing the data samples and their associated emotion vectors.
    n_fft : int, optional
        The number of FFT points used in the STFT calculation. Default is 1024.
    hop_length : int, optional
        The number of samples between successive STFT columns. Default is 512.
    return_spec_shape : bool, optional
        Specifies whether to return the shape of the resulting spectrograms. Default is True.

    Returns
    -------
    spec_df : pandas DataFrame
        DataFrame with the calculated spectrograms and their associated emotion vectors.
    spec_shape : tuple
        The shape of the resulting spectrograms, if `return_spec_shape` is True.

    Notes
    -----
    - This function uses librosa to compute power/magnitude spectrograms and mel-spectrograms from the data samples.
    - The calculated spectrograms are stored in the "Spectrogram" column of the returned DataFrame.
    - The associated emotion vectors from the input DataFrame are preserved in the returned DataFrame.
    - By default, the function also returns the shape of the resulting spectrograms as `spec_shape`.

    """
    spec_list = []

    for sample in data["Data Sample"]:
        # Power/magnitude spectrogram
        # Compute STFT magnitude
        stft = librosa.stft(sample, n_fft=n_fft, hop_length=hop_length) # gives us a 2D numpy array of the shape (n_freq_bins, n_time_frames)
        stft_mag, stft_phase = librosa.magphase(stft)  # extracts the magnitudes and the phases and returns as separate matrices; stft_phase is ignored at the moment
        # Convert to dB scale
        stft_mag_db = librosa.amplitude_to_db(stft_mag, ref=np.max)
        # Turn to absolute values
        mag_spec = np.abs(stft_mag_db) 
        
        # Compute the mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(S=mag_spec)

        # Reshape to 3D, where the third dimension is number of channels (here: 1)
        reshaped = np.reshape(mel_spec, (*mel_spec.shape, 1))

        spec_list.append(reshaped)

    print("The shape of the spectrograms: ", reshaped.shape)

    s = pd.DataFrame(list(zip(spec_list)), columns=["Spectrogram"])
    spec_df = pd.concat([s, data["Emotion Vector"]], axis=1)

    if return_spec_shape:
        return spec_df, reshaped.shape
    else:
        return spec_df



def append_coordinate_information(data:pd.DataFrame, return_spec_shape=True):
    """
    Appends coordinate information to the spectrograms in a DataFrame and combines them into images with 3 channels.

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame containing the spectrograms and their associated emotion vectors.
    return_spec_shape : bool, optional
        Specifies whether to return the shape of the combined spectrograms. Default is True.

    Returns
    -------
    spec_df : pandas DataFrame
        DataFrame with the combined spectrograms and their associated emotion vectors.
    combined_shape : tuple
        The shape of the combined spectrograms, if `return_spec_shape` is True.

    Notes
    -----
    - This function appends coordinate information to the spectrograms in the "Spectrogram" column of the input DataFrame.
    - Coordinate grids are created using TensorFlow's linspace and meshgrid functions.
    - The coordinate grids are duplicated for all spectrograms in the DataFrame.
    - The spectrograms are combined with the coordinate grids to form images with 3 channels.
    - The combined spectrograms are stored in the "Spectrogram" column of the returned DataFrame.
    - The associated emotion vectors from the input DataFrame are preserved in the returned DataFrame.
    - By default, the function also returns the shape of the combined spectrograms as `combined_shape`.

    """
    specs = data["Spectrogram"]
    spec_shape = specs[0].shape  #(n_freq_bins, n_time_frames, channels)
    
    height = spec_shape[0]
    width = spec_shape[1]
    # Create coordinate grids
    x_coords = tf.linspace(-1.0, 1.0, width)
    y_coords = tf.linspace(1.0, -1.0, height)
    x_coords, y_coords = tf.meshgrid(x_coords, y_coords)
    n_specs = specs.shape[0]
    # Duplicate them for all spectrograms
    repeated_x_coords = tf.repeat([x_coords], repeats=n_specs, axis=0)  # Now, the shape is (n_specs, n_freq_bins, n_time_frames, channels)
    repeated_y_coords = tf.repeat([y_coords], repeats=n_specs, axis=0)
    repeated_x_coords = tf.expand_dims(repeated_x_coords, -1)  # Now, the shape is (batch_size, n_freq_bins, n_time_frames, channel)
    repeated_y_coords = tf.expand_dims(repeated_y_coords, -1)
    
    # Combine the spectrograms with the grids to form images with 3 channels -> concatenating along the channel dimension
    combined = tf.concat([np.stack(specs.values), repeated_x_coords, repeated_y_coords], axis=-1)   
    
    s = pd.DataFrame(list(zip(combined)), columns=["Spectrogram"])
    spec_df = pd.concat([s, data["Emotion Vector"]], axis=1)
    
    if return_spec_shape:
        return spec_df, combined[0].shape
    else:
        return spec_df
        
        


def store_preprocessed_data(data: pd.DataFrame, path: str):
    """
    Stores preprocessed data as a pickled pandas DataFrame at the specified path.

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame containing the preprocessed data.
    path : str
        The path where the pickled DataFrame will be stored.

    """
    data.to_pickle(path)



def load_spectrograms(path: str):
    """
    Loads the spectrograms from a pickle file. Extracts the shape of the first spectrogram in the dataframe.

    Returns
    -------
    spec_def : pandas dataframe
        columns = "Spectrogram", "Emotion Vector"
    spec_shape : tuple
        the three dimensional shape of the first spectrogram in the dataframe: (number of frequency bins, number of time frames, number of channels)
        
    """
    spec_df = pd.read_pickle(path)
    spec_shape = spec_df["Spectrogram"].iloc[0].shape
    return spec_df, spec_shape








##########################
# Preconfigured pipelines #
##########################

def load_fix_noise_spec_store_cremad(path_source=default_path_raw_cremad, path_store=default_path_preprocessed_cremad):
    df, target_length = load_data(path_source, dataset_name="cremad")
    print("Target length: ", target_length)
    fixed_df = fix_length(df, target_length)
    noisy_df = add_noise(fixed_df)
    spec_df, spec_shape = calculate_spectrograms(noisy_df)
    spec_df, spec_shape = append_coordinate_information(spec_df)
    store_preprocessed_data(spec_df, path_store)


def load_fix_noise_spec_store_emodb(path=default_path_raw_emodb, path_store=default_path_preprocessed_emodb):
    df, target_length = load_data(path, dataset_name="emodb")
    print("Target length: ", target_length)
    fixed_df = fix_length(df, target_length)
    noisy_df = add_noise(fixed_df)
    spec_df, spec_shape = calculate_spectrograms(noisy_df)
    spec_df, spec_shape = append_coordinate_information(spec_df)
    store_preprocessed_data(spec_df, path_store)




if __name__ == "__main__":
    if os.path.realpath(__file__) != os.getcwd():
        os.chdir(os.path.join(os.path.realpath(__file__), os.pardir))
    load_fix_noise_spec_store_cremad()
    load_fix_noise_spec_store_emodb()
    
