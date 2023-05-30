# imports
import os
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.realpath(__file__), "preprocessing_pipeline.py"))
from preprocessing_pipeline import calculate_spectrograms, pad_with_zeros
sys.path.append(os.path.join(os.path.realpath(__file__), "CRNN_LSTM_model.py"))
from CRNN_LSTM_model import CRNN_LSTM
sys.path.append(os.path.join(os.path.realpath(__file__), "Semantic_approach.py"))
from Semantic_approach import Semantic_Approach
sys.path.append(os.path.join(os.path.realpath(__file__), "gui_audio.py"))
import gui_audio
import threading
import time
import numpy as np
from datetime import datetime

phono_model = CRNN_LSTM(model_name="trial_data_model")  # phonological approach
#lingu_models = Semantic_Approach()  # linguistic approach (semantic = meaning of the words)

ui = gui_audio.VoiceRecorder()

lock = threading.Lock()

data_to_process = []            # Queue for input data that will be fed into the models
data_with_feedback = pd.DataFrame(columns=["Spectrogram", "Feedback"])   # Storage for the samples to train the models later



# Accessed by User Interface
def add_data_to_queue(data):
    print("Appended a recording")
    data_to_process.append(data)
    print(data_to_process)
    pipeline(data_to_process)



# Accessed by the User Interface
def store_feedback(feedback, id):
    data_with_feedback.at[id, "Feedback"] = feedback
    


def pipeline(data_queue):
    print("Listening to input")

    global data_to_process
    global data_with_feedback

    data_sample = data_queue.pop(0)

    data_sample_new = np.nan_to_num(data_sample, copy=True, nan=0.0, posinf=None, neginf=None)


    if len(data_sample_new) < 80000:# Check if data is shorter than 5 sec, TODO: find out the exact number, length = number of frames ?
        data_sample_new = pad_with_zeros(data_sample_new, 80000)
    print(data_sample_new)
    spec = calculate_spectrograms(data_sample_new)

    print("und weiter gehts")

    # Store the spectrogram in a Dataframe for adding the user feedback later. The index of the row is the ID of the sample.
    list_to_append = pd.DataFrame([spec, None])
    data_with_feedback = pd.concat([data_with_feedback, list_to_append])
    id = data_with_feedback.shape[0] - 1  # get the number of rows to compute the index of the last appended row which is the ID of the sample/spectrogram -> TODO: pass it to the GUI, so that it can return the feedback together with the ID of the sample -> necessary for storing the feedback together with the spectrogram
    print("irgendwo hier haperts")
    start_time = datetime.now()
    print(start_time)
    # Prediction based on phonological info
    probs, prediction_1 = phono_model.predict(spec,single_sample=True)  # TODO: maybe also show the predicted probability in the GUI?
    print("nach phono model prediction, vor lingu models prediction")
    print(datetime.now())
    # Prediction based on linguistic info
    #prediction_2 = lingu_models.speech_to_emotion(data_sample)
    prediction_2 = "placeholder"
    print("nach prediction, vor publishing")
    print(datetime.now())
    #print(datetime.now() - start_time)
    ui.publish_emotion_label(prediction_1, prediction_2)









if __name__ == "__main__":


    print("Models have been initialized")

    # Start a thread for the pipeline
    #pipeline = threading.Thread(target=pipeline, daemon=True, args=(data_to_process,))  # daemon -> so that it ends automatically when the GUI is closed
    #pipeline.start()

    
    #print("VoiceRecoder has been initialized")
    ui.launch()


    


        
        




