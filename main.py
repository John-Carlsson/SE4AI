# imports
from tkinter import *
import os
import pandas as pd
import sys

sys.path.append(os.path.join(os.path.realpath(__file__), "preprocessing_pipeline_ser.py"))
from preprocessing_pipeline_ser import preprocess

sys.path.append(os.path.join(os.path.realpath(__file__), "model_ser.py"))
from model_ser import Model

sys.path.append(os.path.join(os.path.realpath(__file__),  "gui_combined.py"))

from gui_combined import App

# sys.path.append(os.path.join(os.path.realpath(__file__), "Semantic_approach.py"))
# from Semantic_approach import Semantic_Approach
sys.path.append(os.path.join(os.path.realpath(__file__), "gui_audio.py"))
import threading
import time
import numpy as np
from datetime import datetime


import cv2
import keras
import numpy as np
from PIL import Image, ImageTk
import platform
import face_detection as fd
import data_collector as dc
from functools import partial
from keras.optimizers import Adam


#phono_model = keras.models.load_model("coord_cnn_gru.pb")
phono_model = Model(model_name="trial_data_model", input_shape=(128, 157, 3))  # phonological approach
# lingu_models = Semantic_Approach()  # linguistic approach (semantic = meaning of the words)


data_to_process = []  # Queue for input data that will be fed into the models
data_with_feedback = pd.DataFrame(
    columns=["Spectrogram", "Feedback"])  # Storage for the samples to train the models later



# Accessed by User Interface
def add_data_to_queue(self, data):
    print("Appended a recording")
    data_to_process.append(data)
    print(data_to_process)
    pipeline(self, data_to_process)


# Accessed by the User Interface
def store_feedback(feedback, id):
    data_with_feedback.at[id, "Feedback"] = feedback


def pipeline(self, data_queue):
    print("Listening to input")

    global data_to_process
    global data_with_feedback

    data_sample = data_queue.pop(0)

    data_sample_new = np.nan_to_num(data_sample, copy=True, nan=0.0, posinf=None, neginf=None)

    spec = preprocess(data_sample_new)

    # Store the spectrogram in a Dataframe for adding the user feedback later. The index of the row is the ID of the sample.
    list_to_append = pd.DataFrame([spec, None])
    data_with_feedback = pd.concat([data_with_feedback, list_to_append])
    id = data_with_feedback.shape[
             0] - 1  # get the number of rows to compute the index of the last appended row which is the ID of the sample/spectrogram -> TODO: pass it to the GUI, so that it can return the feedback together with the ID of the sample -> necessary for storing the feedback together with the spectrogram

    probabilities_ser = phono_model.predict(spec)



    from gui_combined import publish_emotion_label
    publish_emotion_label(self, probabilities_ser)





def combine_results(fer_result: np.array, ser_result: np.array):

    weight_fer = 0.5
    emotion_mapping = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Neutral']
    """ Combines the results of the two individual models, applying a weight on each of the result.

    Parameters
    ----------
    fer_result : np.array
        The probabilities predicted by the FER model
    ser_result : np.array
        The probabilities predicted by the SER model
    weight_fer : float
        The weighting factor by which the FER probabilities are multiplied. The SER probabilities are multiplied by the inverse.
    emotion_mapping : list
        The emotion labels in a specified order

    Returns
    -------
    combined_result : string
        a string representing the most likely emotion

    """

    weighted_fer_result = np.multiply(fer_result, weight_fer)
    weighted_ser_result = np.multiply(ser_result, (1 - weight_fer))

    combined_result = np.add(weighted_fer_result, weighted_ser_result)
    most_likely_index = np.argmax(combined_result, axis=0)
    most_likely_label = emotion_mapping[most_likely_index]

    print(most_likely_label)

    return most_likely_label


if __name__ == "__main__":
    root = Tk()
    model = keras.models.load_model('./sequential_model_c.h5')
    app = App(root, model)
    root.mainloop()
    app.release()



