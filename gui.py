from tkinter import *
import cv2
import keras
import numpy as np
from PIL import Image, ImageTk
import platform



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
        self.camera = Camera(400, 400)

        self.label_widget = Label(self.master)
        self.label_widget.pack(side='left')

        self.good_button = Button(self.master, text='Correct', font=('arial', 25), fg='green', command=self.good_feedback)
        self.bad_button = Button(self.master, text='False', font=('arial', 25), fg='red', command=self.bad_feedback)
        self.capture_button = Button(self.master, text='Capture', font=('arial', 25), fg='black', command=self.run_analysis)
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
        self.detect_face()
        self.show_video()

        self.analyse_face()


    def release(self):
        self.camera.release()

    
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
            face_roi = self.current_frame[y:y+h, x:x+w]

            # Display picture with a frame around the face
            self.current_frame = cv2.rectangle(self.current_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # Convert the region to (48 x 48) grayscale and save the face
            dim = (48, 48)
            self.face = cv2.resize(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY), dim, interpolation = cv2.INTER_AREA)
        else:
            self.text.insert(END, "No face detected\n")

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

    def analyse_face(self):
        img_batch = np.expand_dims(self.face, axis=0)
        result = self.model.predict(np.asarray(img_batch))
        print("result = ", result)
        print(self.to_string(result))
        # result = self.model(self.face)
        self.text.insert(END, self.text.insert(END, self.to_string(result) + '\n'))

if __name__ == '__main__':
    root = Tk()
    app = App(root)
    root.mainloop()
    app.release()

    