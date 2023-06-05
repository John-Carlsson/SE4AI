import os
import sys
import tensorflow as tf
sys.path.append(os.path.join(os.path.realpath(__file__), "preprocessing_training_ser.py"))
from preprocessing_training_ser import load_spectrograms
sys.path.append(os.path.join(os.path.realpath(__file__), "model_ser.py"))
from model_ser import Model


### Default paths ###
# Raw data
default_path_raw_cremad = os.path.join(os.pardir, "data", "cremad")
default_path_raw_emodb = os.path.join(os.pardir, "data", "emodb")
# Preprocessed data
default_path_preprocessed_cremad = os.path.join(os.pardir, "data", "preprocessed_cremad.pickle")
default_path_preprocessed_emodb = os.path.join(os.pardir, "data", "preprocessed_emodb.pickle")



if __name__ == "__main__":
    if os.path.realpath(__file__) != os.getcwd():
        os.chdir(os.path.join(os.path.realpath(__file__), os.pardir))
    print(tf.config.list_physical_devices('GPU'))
    with tf.device('/device:GPU:0'):
    #with tf.device('/CPU:0'):  # if you don't have a gpu
        spec_df, spec_shape = load_spectrograms(path=default_path_preprocessed_cremad)
        print("before creating model")
        model = Model(input_shape=spec_shape)
        print("before training")
        model.train_model(spec_df, epochs=300)
        print("before storing")
        model.store_model()


