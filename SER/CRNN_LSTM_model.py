# imports 
import tensorflow as tf
import keras
from keras import layers
import os
from sklearn.model_selection import train_test_split
import numpy as np

# Default class mapping 
emotion_hot_encode = {
    'ANG': (1., 0., 0., 0., 0., 0.),
    'DIS': (0., 1., 0., 0., 0., 0.),
    'FEA': (0., 0., 1., 0., 0., 0.),
    'HAP': (0., 0., 0., 1., 0., 0.),
    'NEU': (0., 0., 0., 0., 1., 0.),
    'SAD': (0., 0., 0., 0., 0., 1.)
}

class CRNN_LSTM:

    def __init__(self, model_name="Default", class_mapping=emotion_hot_encode, input_shape=(513, 157, 1), path="Models", model=None):
        self.model_name = model_name
        self.class_mapping = class_mapping
        self.input_shape = input_shape
        self.path = path
        self.model = model

        if self.model == None:
            if os.path.isfile(os.path.join(self.path, self.model_name, ".h5")):
                self.model = self._load_model()
            else: 
                self.model = self._create_model()





    def _load_model(self):
        self.model = tf.keras.models.load_model(os.path.join(self.path, self.model_name, ".h5"))
        return self.model
    



    def _create_model(self):
        self.model = keras.Sequential()

        ### CNN ###
        # First module
        self.model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='elu', input_shape=self.input_shape))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        # Second module
        self.model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='elu'))
        self.model.add(layers.BatchNormalization()) # maybe remove this layer, as it could lead to overfitting (but could also improve performance)
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        # Third module
        self.model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='elu'))
        self.model.add(layers.BatchNormalization())
        last_cnn_layer = layers.MaxPooling2D(pool_size=(2, 2))
        self.model.add(last_cnn_layer)

        ### Connection ###
        time_steps = -1 # is derived by keras
        features_maps = last_cnn_layer.output_shape[3] # number of feature maps
        self.model.add(layers.Reshape((time_steps, features_maps)))
        
        ### LSTM ###
        self.model.add(layers.LSTM(units=64, activation='tanh', return_sequences=True)) # time_steps = 64, typical range from 64 to 512
        self.model.add(layers.LSTM(units=32, activation='tanh'))

        ### Output ###
        self.model.add(layers.Dense(units=len(self.class_mapping), activation='softmax')) 

        ### Compilation ###
        opt = keras.optimizers.SGD(lr=0.1, momentum=0.9) # experiment with learning rate
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


        self.model.summary()


        return self.model



    ## Dataframe Columns: "Data Sample", "Name", "Emotion Class", "Emotion Vector", "Padded Sample", "Spectrogram"
    def train_model(self, spec_data, epochs=50):
        X_train, X_test, y_train, y_test = train_test_split(spec_data['Spectrogram'], spec_data['Emotion Vector'], test_size=0.2, random_state=0)

        # Convert the samples and the labels to Tensor
        X_train_tensors = tf.convert_to_tensor([tf.convert_to_tensor(sample) for sample in X_train])
        X_train_tensors = np.reshape(X_train_tensors, (*X_train_tensors.shape, 1))
        y_train_tensors = tf.convert_to_tensor([tf.convert_to_tensor(label) for label in y_train])


        X_test_tensors = tf.convert_to_tensor([tf.convert_to_tensor(sample) for sample in X_test])
        X_test_tensors = np.reshape(X_test_tensors, (*X_test_tensors.shape, 1))
        y_test_tensors = tf.convert_to_tensor([tf.convert_to_tensor(label) for label in y_test])


        self.model.fit(X_train_tensors, y_train_tensors, epochs=epochs) # number epochs need to be examined

        loss, accuracy = self.model.evaluate(X_test_tensors, y_test_tensors)
        return loss, accuracy




    def predict(self, data, single_sample=True):
        if single_sample:
            return self.model(data)
        else:
            return self.model.predict(data)
        
        


    def store_model(self):
        self.model.save(self.path)
        print("Model %s has been stored in %s. "%(self.model_name, self.path))


if __name__ == "__main__":
    model = CRNN_LSTM()
    model.store_model()