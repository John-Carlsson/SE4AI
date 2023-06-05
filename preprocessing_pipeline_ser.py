import librosa
import numpy as np
import tensorflow as tf



def preprocess(input_sample, target_length=80000):
    """
    Preprocesses an input audio sample by fixing the length, transforming it to a mel spectrogram 
    and appending coordinate information to each pixel of the spectrogram.

    Parameters
    ----------
    input_sample : numpy.ndarray
        Input audio sample as a 1-dimensional numpy array.
    target_length : float, optional
        Target length of the processed sample in seconds. Defaults to 80000 samples.

    Returns
    -------
    numpy.ndarray
        Preprocessed audio sample, transformed into a 3-dimensional numpy array.

    """
    # Adjust the length of the input sample
    fixed_size_sample = librosa.util.fix_length(input_sample, size=int(target_length))

    # Power/magnitude spectrogram
    # Compute STFT magnitude
    stft = librosa.stft(fixed_size_sample, n_fft=1024, hop_length=512) # gives us a 2D numpy array of the shape (n_freq_bins, n_time_frames)
    stft_mag, _ = librosa.magphase(stft)  # extracts the magnitudes and the phases and returns as separate matrices; stft_phase is ignored at the moment
    # Convert to dB scale
    stft_mag_db = librosa.amplitude_to_db(stft_mag, ref=np.max)
    # Turn to absolute values
    mag_spec = np.abs(stft_mag_db) 
    
    # Compute the mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(S=mag_spec)

    # Reshape to 3D, where the third dimension is number of channels (here: 1)
    reshaped_spec = np.reshape(mel_spec, (*mel_spec.shape, 1))

    height = reshaped_spec.shape[0]
    width = reshaped_spec.shape[1]
    # Create coordinate grids
    x_coords = tf.linspace(-1.0, 1.0, width)
    y_coords = tf.linspace(1.0, -1.0, height)
    x_coords, y_coords = tf.meshgrid(x_coords, y_coords)
    # Add channel dimension
    expanded_x_coords = tf.expand_dims(x_coords, -1)
    expanded_y_coords = tf.expand_dims(y_coords, -1)

    # Append the coordinate information to the spectrogram 
    combined = tf.concat([reshaped_spec, expanded_x_coords, expanded_y_coords], axis=-1)

    return combined

