import librosa
import numpy as np



def pad_with_zeros(data, target_length=None):
    padded_data = librosa.util.pad_center(data, size=target_length)
    return padded_data



def calculate_spectrograms(data, n_fft=1024, hop_length=512):
    # Compute STFT magnitude
    stft = librosa.stft(data, n_fft=n_fft, hop_length=hop_length) # gives us a 2D numpy array of the shape (n_freq_bins, n_time_frames)
    stft_mag, stft_phase = librosa.magphase(stft)  # extracts the magnitudes and the phases and returns as separate matrices; stft_phase is ignored at the moment
    # Convert to dB scale
    stft_mag_db = librosa.amplitude_to_db(stft_mag, ref=np.max)
    # Turn to absolute values
    mag_spec = np.abs(stft_mag_db) 
    # Reshape to 3D, where the third dimension is number of channels (here: 1)
    reshaped_spec = np.reshape(mag_spec, (*mag_spec.shape, 1))
    
    return reshaped_spec