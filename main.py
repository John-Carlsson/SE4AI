# imports
from tkinter import *
import os
import pandas as pd
import sys

sys.path.append(os.path.join(os.path.realpath(__file__), "SER", "Dagshub_code", "preprocessing_pipeline_ser.py"))
from preprocessing_pipeline_ser import preprocess

sys.path.append(os.path.join(os.path.realpath(__file__), "SER", "Dagshub_code", "model_ser.py"))
#from model_ser import Model

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


#phono_model = Model(model_name="trial_data_model", input_shape=(128, 157, 3))  # phonological approach
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


if __name__ == "__main__":
    print(os.getcwd())
    root = Tk()
    model = keras.models.load_model('./sequential_model_c.h5')
    app = App(root, model)
    root.mainloop()
    app.release()



