# imports
import os
import sys
sys.path.append(os.path.join(os.path.realpath(__file__), "preprocessing_ser.py"))
from preprocessing_ser import load_pad_spec_store, load_spectrograms
sys.path.append(os.path.join(os.path.realpath(__file__), "CRNN_LSTM_model.py"))
from CRNN_LSTM_model import CRNN_LSTM



if __name__ == "__main__":
    load_pad_spec_store("Trial_Data")
    specs = load_spectrograms()
    spec_shape = (specs["Spectrogram"].iloc[0].shape[0], specs["Spectrogram"].iloc[0].shape[1], 1)
    model = CRNN_LSTM(model_name="trial_data_model", input_shape=spec_shape)
    #model.train_model(specs)
    model.store_model()
    #pred = model.predict(specs["Spectrogram"].iloc[0])
    #print(pred)