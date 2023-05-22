# imports
import os
import sys
sys.path.append(os.path.join(os.path.realpath(__file__), "preprocessing_ser.py"))
from preprocessing_ser import load_pad_spec_store, load_spectrograms
sys.path.append(os.path.join(os.path.realpath(__file__), "CRNN_LSTM_model.py"))
from CRNN_LSTM_model import CRNN_LSTM
sys.path.append(os.path.join(os.path.realpath(__file__), "Semantic_approach.py"))
from Semantic_approach import Semantic_Approach



if __name__ == "__main__":
    # Input
    load_pad_spec_store("Trial_Data")
    specs, spec_shape = load_spectrograms()
    print(specs)
    print(spec_shape)



    # First approach
    model = CRNN_LSTM(model_name="trial_data_model", input_shape=spec_shape)
    #model.train_model(specs)
    #model.store_model()
    probs, em_labels = model.predict(specs["Spectrogram"].iloc[0], single_sample=True)
    print(probs)
    print(em_labels)



    # Second approach
    sem_appr = Semantic_Approach()
    transcription = sem_appr.speech_to_text(specs["Padded Sample"].iloc[0])
    print(transcription)
    prediction = sem_appr.text_to_emotion(transcription)
    print(prediction)