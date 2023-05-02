import os
import numpy as np
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
#import file_import
from skimage import io
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
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import LeakyReLU

import nnmodel


matplotlib.use('TkAgg')

#path1 = '/Users/psleborne/Documents/Vorlesungen/Skript/SoftwareEngAI/Emotional_Recognition/Datasets/data1'
#path2 = '/Users/psleborne/Documents/Vorlesungen/Skript/SoftwareEngAI/Emotional_Recognition/Datasets/data2'


#data = pd.read_csv('/Users/psleborne/Documents/Vorlesungen/Skript/SoftwareEngAI/Emotional_Recognition/Datasets/data2/fer2013.csv')
#emotion_map = {0: 'Angry', 1: 'Digust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
#emotion_counts = data['emotion'].value_counts(sort=False).reset_index()
#emotion_counts.columns = ['emotion', 'number']
#emotion_counts['emotion'] = emotion_counts['emotion'].map(emotion_map)
#print(emotion_counts)

emotion_labels = ['Angry', 'disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# initilize parameters
num_classes = 7
width, height = 48, 48
num_epochs = 50
batch_size = 64
num_features = 64


def row2image(row):
    pixels, emotion = row['pixels'], emotion_map[row['emotion']]
    img = np.array(pixels.split())
    img = img.reshape(48, 48)
    image = np.zeros((48, 48, 3))
    image[:, :, 0] = img
    image[:, :, 1] = img
    image[:, :, 2] = img
    return np.array([image.astype(np.uint8), emotion])


def main():
    # print("Using Tensorflow version: ", tf.__version__)
    # Tensorflow: create a TF dataset from folder

    # For performance testing: (This disables the usage of the GPU)
    # tf.config.set_visible_devices([], 'GPU')

    #print(os.listdir(path2))

    # check data shape
    #data.shape
    #data.head(5)
    #data.Usage.value_counts()

    # Plotting a bar graph of the class distributions
    #plt.figure(figsize=(6, 4))
    #sns.barplot(x=emotion_counts.emotion, y=emotion_counts.number)
    #plt.title('Class distribution')
    #plt.ylabel('Number', fontsize=12)
    #plt.xlabel('Emotions', fontsize=12)
    #plt.show()

    #plt.figure(0, figsize=(16, 10))
    #for i in range(1, 8):
    #    face = data[data['emotion'] == i - 1].iloc[0]
    #    img = row2image(face)
    #    plt.subplot(2, 4, i)
    #    plt.imshow(img[0])
    #    plt.title(img[1])

    #plt.show()

    # new_model = nnmodel.NNmodel('sequential', num_classes, num_epochs, batch_size, num_features, width, height)
    # model_path = new_model.train_model()
    # print(model_path)

##### Tresting model: #########

    new_model = nnmodel.NNmodel('sequential1', num_classes, num_epochs, batch_size, num_features, width, height)

    model_path = './sequential_model_c.h5'

    model = new_model.nload_model(model_path)
    model.summary()
    #Testing model on test-dataset
    train_X, train_Y, val_X, val_Y, test_X, test_Y = new_model.process_split_data()
    test_loss, test_acc = model.evaluate(test_X, test_Y)
    print(test_acc)
    #selecting a random image from test-set
    selected_img = np.random.randint(0, 3589)
    features = test_X[selected_img]
    img = np.array(features)
    img = img.reshape(48, 48)
    image1 = np.zeros((48, 48, 3))
    image1[:, :, 0] = img * 255
    image1[:, :, 1] = img * 255
    image1[:, :, 2] = img * 255
    plt.imshow(image1.astype(np.uint8))
    plt.show()
    label = test_Y[selected_img]
    print(label)
    print(features)
    print(np.asarray(features).shape)
    img_array = tf.keras.utils.img_to_array(features)
    img_batch = np.expand_dims(img_array, axis=0)
    prediction = model.predict(np.asarray(img_batch))
    print("Actual:", label )
    print("Prediction:", prediction)

    #Running the model on a captured image:
    img_path = '/Users/psleborne/IdeaProjects/SE4AI/image_preprocessing/face0.jpg'
    # plt.imshow(img_path)
    # plt.show()

    img1 = tf.keras.utils.load_img(img_path, color_mode = "grayscale", target_size=(48, 48))
    img_array = tf.keras.utils.img_to_array(img1)
    img_batch = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_batch)
    print(prediction)



if __name__ == "__main__":
    main()
