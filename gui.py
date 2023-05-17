import os
from tkinter import *
import cv2
import keras
import numpy as np
from PIL import Image, ImageTk
import platform
import data_collector as dc
import pyautogui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd


# from PyQt5 import QtCore, QtWidgets


class Camera:
    def __init__(self, width, height):
        if platform.processor() == 'arm':
            self.vid = cv2.VideoCapture(1)
        else:
            self.vid = cv2.VideoCapture(0)
        self.width = width
        self.height = height
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def capture_frame(self):
        _, frame = self.vid.read()
        return cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA), 1)

    def release(self):
        self.vid.release()


class App:

    def __init__(self, master):
        self.master = master
        self.model = keras.models.load_model('./sequential_model_c.h5')
        self.current_frame = None
        self.master.bind('<Escape>', lambda e: self.master.quit())
        self.width, self.height = pyautogui.size()
        self.camera = Camera(self.width / 5, self.height / 5)
        self.analyse = False

        self.angry = 0.0
        self.disgust = 0.0
        self.fear = 0.0
        self.happy = 50.0
        self.sad = 0.0
        self.surprised = 0.0
        self.neutral = 0.0

        self.label_widget = Label(self.master)
        self.label_widget.pack(side='left')

        self.data = {'emotions': ['Angry', 'disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
                     'predictions': [self.angry, self.disgust, self.fear, self.happy, self.sad, self.surprised,
                                     self.neutral]}
        self.figure1 = plt.Figure(figsize=(6, 5), dpi=100)
        self.df = pd.DataFrame(self.data)

        self.ax1 = self.figure1.add_subplot(111)
        self.bar1 = FigureCanvasTkAgg(self.figure1, root)
        self.bar1.get_tk_widget().pack(side='left', fill='both')
        self.plot_ref = False
        self.ax1.set_xticks([0, 1, 2, 3, 4, 5, 6])
        self.ax1.set_xticklabels(['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
                                 rotation='vertical', fontsize=8)
        self.ax1.set_title('Predictions')
        self.ax1.set_ylim([0, 100])

        self.good_button = Button(self.master, text='Correct', font=('arial', 25), fg='green',
                                  command=self.good_feedback)
        self.bad_button = Button(self.master, text='False', font=('arial', 25), fg='red', command=self.bad_feedback)
        self.capture_button = Button(self.master, text='Capture', font=('arial', 25), fg='black',
                                     command=self.run_analysis)
        self.capture_button.pack(side='bottom')
        self.bad_button.pack(side='bottom')
        self.good_button.pack(side='bottom')

        self.text = Text(self.master, height=20, width=100, bg='skyblue')
        self.text.pack(side='top')
        self.face = None

        self.show_video()

    def show_video(self):
        if self.current_frame is not None:  # if we have a captured frame, display it
            captured_image = Image.fromarray(self.current_frame)
            photo_image = ImageTk.PhotoImage(image=captured_image)
            self.label_widget.photo_image = photo_image
            self.label_widget.configure(image=photo_image)
        else:  # otherwise, display the current camera frame
            opencv_image = self.camera.capture_frame()
            captured_image = Image.fromarray(opencv_image)
            photo_image = ImageTk.PhotoImage(image=captured_image)
            self.label_widget.photo_image = photo_image
            self.label_widget.configure(image=photo_image)
            self.label_widget.after(10, self.show_video)

    def good_feedback(self):
        self.current_frame = None  # Disable stillframe
        self.show_video()

    def bad_feedback(self):
        self.current_frame = None  # Disable stillframe
        self.show_video()

    def run_analysis(self):
        self.current_frame = self.camera.capture_frame()  # capture a frame
        self.detect_face2()
        self.show_video()
        if self.analyse:
            self.analyse_face()

    def release(self):
        self.camera.release()


    def detect_face2(self):
        # Define paths
        prototxt_path = os.path.join('image_preprocessing/deploy.prototxt')
        caffemodel_path = os.path.join('image_preprocessing/weights.caffemodel')

        # Read the model
        model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

        image = self.current_frame

        cv2.imwrite("imageH.jpg", image)

        image = cv2.imread('imageH.jpg')

        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        model.setInput(blob)
        detections = model.forward()

        # Identify each face
        for i in range(0, detections.shape[2]):
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            confidence = detections[0, 0, i, 2]

            # If confidence > 0.5, save it as a separate file
            if (confidence > 0.5):
                frame = image[startY:endY, startX:endX]
                self.current_frame = cv2.rectangle(self.current_frame, (startX, startY), (endX, endY), (0, 255, 255), 2)
                dim = (48, 48)
                self.face = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dim, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(i) + '_' + "image.jpg", self.face)

    def detect_face(self):
        # Convert the image to grayscale
        gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)

        # Load the Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier('image_preprocessing/haarcascade_frontalface_alt.xml')

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)

        if len(faces) > 0:
            # Get the first detected face So that if multiple persons in frame we just look at one
            (x, y, w, h) = faces[0]

            # Extract the region of interest containing the face
            face_roi = self.current_frame[y:y + h, x:x + w]

            # Display picture with a frame around the face
            self.current_frame = cv2.rectangle(self.current_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

            n = dc.collector()
            # TODO: emotion input, default is set to 1
            e = 1

            # Convert the region to (48 x 48) grayscale and save the face
            dim = (48, 48)
            self.face = cv2.resize(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY), dim, interpolation=cv2.INTER_AREA) / 255
            n.save_img(self.face, e)
            cv2.imwrite('./face_test.png', self.face)
            # print(self.width, self.height)
            self.analyse = True
        else:
            self.text.insert(END, "No face detected\n")
            self.analyse = False

    def to_string(self, result):
        emotion_labels = ['Angry', 'disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        s = ''
        plist = result[0]
        for i in range(7):
            p = round(plist[i] * 100, 3)
            s += emotion_labels[i] + ': '
            s += str(p) + '%'
            s += '\n'
        return s

    def update_predictions(self, results):
        self.angry = results[0]
        self.disgust = results[1]
        self.fear = results[2]
        self.happy = results[3]
        self.sad = results[4]
        self.surprised = results[5]
        self.neutral = results[6]

    def update_plot(self):
        self.data = {'emotions': ['Angry', 'disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
                     'predictions': [self.angry, self.disgust, self.fear, self.happy, self.sad, self.surprised,
                                     self.neutral]}
        self.df = pd.DataFrame(self.data)

        # TODO: This may not be optimal, but it works at updating the frame after each capture.
        if not self.plot_ref:
            self.plot_ref = True
            self.df.plot(kind='bar', legend=False, ax=self.ax1)

        else:
            self.ax1.clear()
            self.df.plot(kind='bar', legend=False, ax=self.ax1)

        # Trigger the canvas to update and redraw.
        self.ax1.set_xticks([0, 1, 2, 3, 4, 5, 6])
        self.ax1.set_xticklabels(['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
                                 rotation='vertical', fontsize=8)
        self.ax1.set_title('Predictions')
        self.ax1.set_ylim([0, 100])
        self.bar1.draw()


    def analyse_face(self):
        img_batch = np.expand_dims(self.face, axis=0)
        result = self.model.predict(np.asarray(img_batch))
        self.update_predictions(result[0] * 100)
        self.update_plot()
        print("result = ", result)
        print(self.to_string(result))
        # result = self.model(self.face)
        self.text.insert(END, self.text.insert(END, self.to_string(result) + '\n'))


if __name__ == '__main__':
    root = Tk()
    app = App(root)
    root.mainloop()
    app.release()

    