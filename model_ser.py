import tensorflow as tf
import keras
from keras import layers
from keras import callbacks
import os
from sklearn.model_selection import train_test_split
import numpy as np
import mlflow
import pandas as pd
import time
import json
import sys
sys.path.append(os.path.join(os.path.realpath(__file__), "user_data.py"))
#from user_data import DAGSHUB_TOKEN, DAGSHUB_USER_NAME


#os.environ['MLFLOW_TRACKING_URI']=f"https://dagshub.com/{DAGSHUB_USER_NAME}/SER.mlflow"

# Recommended to define as environment variables
#os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USER_NAME
#os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN




### Default path ###
#default_path_store_model = os.path.join(os.pardir, "data")
default_path_store_model = ""
default_path_checkpoint_weights = os.path.join(os.pardir, "data", "cp_weights")

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


class Model:

    def __init__(self, model_name: str="model", number_classes: int=6, input_shape: tuple=(1, 1, 1), path: str=default_path_store_model, model: tf.keras.Model=None):
        self.model_name = model_name
        self.number_classes = number_classes
        self.input_shape = input_shape
        self.path = path
        self.model = model

        if self.model == None:
            if os.path.isfile("./coord-cnn-gru.h5"):
                self.model = self._load_model()
                print("Model %s has been loaded from %s. "%(self.model_name, self.path))
            else: 
                self.model = self._create_model()


    def _load_model(self):
        self.model = tf.keras.models.load_model("./coord-cnn-gru.h5")
        return self.model
    

    def _create_model(self):
        self.model = keras.Sequential()

        ### CNN ###
        # First module
        self.model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='elu', input_shape=self.input_shape))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        # Second module
        self.model.add(layers.Dropout(0.4))
        self.model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='elu'))
        self.model.add(layers.BatchNormalization()) # maybe remove this layer, as it could lead to overfitting (but could also improve performance)
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        # Third module
        self.model.add(layers.Dropout(0.3))
        self.model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='elu'))
        self.model.add(layers.BatchNormalization())
        last_cnn_layer = layers.MaxPooling2D(pool_size=(2, 2))
        self.model.add(last_cnn_layer)

        ### Connection ###
        time_steps = -1 # is derived by keras
        feature_maps = last_cnn_layer.output_shape[3] # number of feature maps
        self.model.add(layers.Reshape((time_steps, feature_maps)))
        
        ### LSTM ###
        self.model.add(layers.GRU(units=128, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, reset_after=True)) # units = time_steps, typical range from 64 to 512
        self.model.add(layers.GRU(units=64, activation='tanh', recurrent_activation='sigmoid', reset_after=True)) 
        #self.model.add(layers.Dropout(0.2))

        ### Dense ###
        self.model.add(layers.Dense(units=128, activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(units=64, activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.5))

        ### Output ###
        self.model.add(layers.Dense(units=self.number_classes, activation='softmax')) 

        ### Compilation ###
        #opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9) # experiment with learning rate
        opt = keras.optimizers.Adam(learning_rate=0.001) 
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


        self.model.summary()


        return self.model
    




    ## Dataframe Columns: "Data Sample", "Name", "Emotion Class", "Emotion Vector", "Padded Sample", "Spectrogram"
    def train_model(self, spec_data, epochs=50, checkpoint_path=default_path_checkpoint_weights):
        run_id = mlflow.start_run().info.run_id

        # Measure training time
        start_time = time.time()
        X_train, X_test, y_train, y_test = train_test_split(spec_data['Spectrogram'], spec_data['Emotion Vector'], test_size=0.2, random_state=0)
        #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
        print("Model will be trainend on %i training samples, on %i validating samples and on %i testing samples."%(len(X_train)*0.75, len(X_train)*0.25, len(X_test)))


        # Convert the samples and the labels to the type Tensor
        X_train_tensors = tf.convert_to_tensor([tf.convert_to_tensor(sample) for sample in X_train])
        X_train_tensors = np.reshape(X_train_tensors, (*X_train_tensors.shape, 1))
        y_train_tensors = tf.convert_to_tensor([tf.convert_to_tensor(label) for label in y_train])

        #X_val_tensors = tf.convert_to_tensor([tf.convert_to_tensor(sample) for sample in X_val])
        #X_val_tensors = np.reshape(X_val_tensors, (*X_val_tensors.shape, 1))
        #y_val_tensors = tf.convert_to_tensor([tf.convert_to_tensor(label) for label in y_val])

        X_test_tensors = tf.convert_to_tensor([tf.convert_to_tensor(sample) for sample in X_test])
        X_test_tensors = np.reshape(X_test_tensors, (*X_test_tensors.shape, 1))
        y_test_tensors = tf.convert_to_tensor([tf.convert_to_tensor(label) for label in y_test])

        es = callbacks.EarlyStopping(monitor='loss', patience=10, mode='min', restore_best_weights=True)
        best_weights = callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, save_weights_only=True)

        mlflow.tensorflow.autolog(log_models=True, registered_model_name=self.model_name, log_model_signatures=True)

        self.model.fit(X_train_tensors, y_train_tensors, epochs=epochs, callbacks=[es, best_weights], validation_split=0.25, use_multiprocessing=True) # number epochs need to be examined
        
        
        """
        #model_uri = mlflow.get_artifact_uri("cnn-lstm")
        #model_uri = mlflow.tensorflow.log_model(self.model, "cnn-lstm").model_uri
        print(X_test, y_test)
        mlflow_test = pd.concat([X_test, y_test], axis=1)
        print(mlflow_test)
        result = mlflow.evaluate(
            model_uri,
            data = mlflow_test,
            targets="Emotion Vector",
            model_type="classifier",
            #dataset_name="crema d",
            evaluators=["default"],
        )
        print(result)
        """

        # End training time measurement
        end_time = time.time()

        #run_id = mlflow.active_run().info.run_id
        autologged_data = mlflow.get_run(run_id=run_id)
        # Create metric for model training time
        with open(os.path.join(os.pardir, "metrics", "train_metric.json"), 'w') as f:
            metrics = {'training_time': end_time - start_time}
            metrics.update(autologged_data.data.metrics)
            json.dump(metrics, f)
        
        with open(os.path.join(os.pardir, "params", "train_param.json"), 'w') as f:
            json.dump(autologged_data.data.params, f)

        loss, accuracy = self.model.evaluate(X_test_tensors, y_test_tensors)
        mlflow.end_run()
        return loss, accuracy



    def load_weights(self, weights_path):
      self.model.load_weights(weights_path)

    def predict(self, data):
        return self.model.predict(tf.convert_to_tensor(data[np.newaxis, ...]))



    def store_model(self):
        self.model.save(os.path.join(self.path, self.model_name + ".h5"), save_format='h5')
        print("Model %s has been stored in %s. "%(self.model_name, self.path))

