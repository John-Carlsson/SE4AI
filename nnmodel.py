import file_import
import tensorflow as tf
import numpy as np
import matplotlib
import pandas as pd
import keras
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from sklearn.metrics import accuracy_score
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import LeakyReLU

from keras.callbacks import ReduceLROnPlateau
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import schedules

import fetch_dataset


########################
# for testing purposes #
########################


#default_path = '/Users/psleborne/Documents/Vorlesungen/Skript/SoftwareEngAI/Emotional_Recognition/Datasets/data2/fer2013.csv'
#path2 = '/Users/psleborne/Documents/Vorlesungen/Skript/SoftwareEngAI/Emotional_Recognition/Datasets/data2'

fetch_dataset.download()
default_path = 'fer2013.csv'


#default_data = pd.read_csv(path2 + '/fer2013.csv')
default_data = pd.read_csv(default_path)
emotion_map_default = {0: 'Angry', 1: 'Digust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
emotion_counts = default_data['emotion'].value_counts(sort=False).reset_index()
emotion_counts.columns = ['emotion', 'number']
emotion_counts['emotion'] = emotion_counts['emotion'].map(emotion_map_default)
emotion_labels_default = ['Angry', 'disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

##########################


data_generator = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=.1,
    horizontal_flip=True
)


class NNmodel:

    def __init__(self, model_name, num_classes, num_epochs, batch_size, num_features, width, height,
                 path=default_path, emotion_map=emotion_map_default, emotion_labels=emotion_labels_default):
        self.model_name = model_name
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_features = num_features
        self.width = width
        self.height = height
        self.path = path
        self.emotion_map = emotion_map
        self.emotion_labels = emotion_labels

    def CRNO(self, df, dataName):
        df['pixels'] = df['pixels'].apply(lambda pixel_sequence: [int(pixel) for pixel in pixel_sequence.split()])
        data_X = np.array(df['pixels'].tolist(), dtype='float32').reshape(-1, self.width, self.height, 1) / 255.0
        data_Y = to_categorical(df['emotion'], self.num_classes)
        print(dataName, "_X shape: {}, ", dataName, "_Y shape: {}".format(data_X.shape, data_Y.shape))
        return data_X, data_Y

    def split_data(self, showfig=False):
        data = pd.read_csv(self.path)
        data_train = data[data['Usage'] == 'Training'].copy()
        data_val = data[data['Usage'] == 'PublicTest'].copy()
        data_test = data[data['Usage'] == 'PrivateTest'].copy()

        return data_train, data_val, data_test

    def plot_splitdata(self, data_train, data_val, data_test):
        print("train shape: {}, \nvalidation shape: {}, \ntest shape: {}".format(data_train.shape, data_val.shape,
                                                                                 data_test.shape))

        fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=True)
        setup_axe(axes[0], data_train, 'train')
        setup_axe(axes[1], data_val, 'validation')
        setup_axe(axes[2], data_test, 'test')
        plt.show()

    def train_model(self):
        model_name = self.model_name
        num_features = self.num_features

        if model_name == 'sequential':
            model = Sequential()
            # module 1
            model.add(
                Conv2D(2 * 2 * num_features, kernel_size=(3, 3), input_shape=(48, 48, 1), data_format='channels_last'))
            model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=0.1))
            model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), padding='same'))
            model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=0.1))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

            # module 2
            model.add(Conv2D(4 * num_features, kernel_size=(3, 3), padding='same'))
            model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=0.1))
            model.add(Conv2D(4 * num_features, kernel_size=(3, 3), padding='same'))
            model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=0.1))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

            # module 3
            model.add(Conv2D(2 * num_features, kernel_size=(3, 3), padding='same'))
            model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=0.1))
            model.add(Conv2D(2 * num_features, kernel_size=(3, 3), padding='same'))
            model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=0.1))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

            # flatten
            model.add(Flatten())

            # dense 2
            model.add(Dense(2 * 2 * num_features))
            model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=0.1))
            model.add(Dropout(0.3))

            # dense 4
            model.add(Dense(2 * num_features))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.1))

            model.add(Dense(7, activation='softmax'))

            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                0.01,
                decay_steps=100,
                decay_rate=0.5,
                staircase=False)

            model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.Adam(0.01),
                          metrics=['accuracy'])

            model.summary()
        else:
            print("The selected model is not available")

        data_train, data_val, data_test = self.split_data()
        train_X, train_Y = self.CRNO(data_train, "train")  # training data
        val_X, val_Y = self.CRNO(data_val, "val")  # validation data
        test_X, test_Y = self.CRNO(data_test, "test")  # test data

        es = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

        cp_path = './model_bacc.h5'
        cpl_path = './model_fer_bloss.h5'
        cp = tf.keras.callbacks.ModelCheckpoint(filepath=cp_path, save_best_only=True, save_weights_only=False,
                                                verbose=0, monitor='val_accuracy')
        cpl = tf.keras.callbacks.ModelCheckpoint(filepath=cpl_path, save_best_only=True, save_weights_only=False,
                                                 verbose=2, monitor='val_loss')
        history = model.fit(data_generator.flow(train_X, train_Y, 256),
                            epochs=100,
                            verbose=1,
                            callbacks=[cp, cpl, es],
                            validation_data=(val_X, val_Y))
        model.save('./' + self.model_name + '_model_c.h5')
        return './model_c.h5'


# CRNO stands for Convert, Reshape, Normalize, One-hot encoding
# (i) convert strings to lists of integers
# (ii) reshape and normalise grayscale image with 255.0
# (iii) one-hot encoding label, e.g. class 3 to [0,0,0,1,0,0,0]




def setup_axe(axe, df, title):
    df['emotion'].value_counts(sort=False).plot(ax=axe, kind='bar', rot=0)
    axe.set_xticklabels(emotion_labels)
    axe.set_xlabel("Emotions")
    axe.set_ylabel("Number")
    axe.set_title(title)

    # set individual bar lables using above list
    for i in axe.patches:
        # get_x pulls left or right; get_height pushes up or down
        axe.text(i.get_x() - .05, i.get_height() + 120, \
                 str(round((i.get_height()), 2)), fontsize=14, color='dimgrey',
                 rotation=0)
