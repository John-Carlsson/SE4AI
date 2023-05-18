from tkinter import *
import cv2
import keras
import numpy as np
from PIL import Image, ImageTk
import platform
import data_collector as dc
from functools import partial
from keras.optimizers import Adam


class Camera:

    def __init__(self, width:int, height:int):
        """ Init of the camera class that can capture a frame

        Args:
            width (int): [set the width of the camera]
            height (int): [set the width of the camera]
        """

        if platform.processor() == 'arm':
            self.vid = cv2.VideoCapture(1)
        else:
            self.vid = cv2.VideoCapture(0)
        self.width = width
        self.height = height
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def capture_frame(self):
        """Capture a frame from the camera.

        Returns:
            numpy.ndarray: Captured frame as an RGB image array.
        """
        _, frame = self.vid.read()
        return cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA), 1)

    def release(self):
        self.vid.release()

class App:
    """The GUI app with three windows: camera, output, and buttons."""

    def __init__(self, master, model, face_model = cv2.CascadeClassifier('image_preprocessing/haarcascade_frontalface_alt.xml')):
        """Initialize the app.

        Args:
            master: The parent window.
            model: A Keras model used for analysis.
            face_model: A opencv model used to identify faces
        """
        self.master = master
        self.model = model
        self.face_detection = face_model
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
        """Display the video stream from the camera."""
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
        """Handle the 'Correct' button click event."""
        if self.current_frame is not None:
            self.feedback = 1
            self.current_frame = None  # Disable stillframe
            self.show_video()
            self.update_model_with_feedback()

    def bad_feedback(self):
        """Handle the 'False' button click event."""
        if self.current_frame is not None:
            self.current_frame = None  # Disable stillframe
            self.show_video()
            self.update_model_with_feedback()
            # Since we dont know what is wrong we can't update with any precise labels
            # this becomes very impractible with more than 2 labels
            # so this is just for possible use in for example a binary classifier
            

    def update_model_with_feedback(self):
        """Update the model with the provided feedback."""
        if self.feedback is not None:
            img_batch = np.expand_dims(self.face, axis=0)
            
            idx = np.where(max(self.result[0]) == self.result[0])
            self.result[0] = [0 for i in range(len(self.result[0]))]
            self.result[0][idx] = 1

            feedback_data = self.result

            optimizer = Adam(lr=0.001)
            self.model.compile(loss='binary_crossentropy', optimizer=optimizer)
            self.model.fit(img_batch, feedback_data, epochs=1)

        self.feedback = None

    # Capture button
    def run_analysis(self):
        """Perform face analysis on the captured frame."""
        self.current_frame = self.camera.capture_frame()  # capture a frame
        self.detect_face()
        self.show_video()
        if self.current_frame is not None:
            self.analyse_face()


    def release(self):
        """Release the camera."""
        self.camera.release()

    
    def detect_face(self):
        """Detect faces in the current frame and extract the region of interest."""
        # Convert the image to grayscale
        gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image and return them
        faces = self.face_detection.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)

        if len(faces) > 0:
            # Get the first detected face So that if multiple persons in frame we just look at one
            (x, y, w, h) = faces[0]

            # Extract the region of interest containing the face
            face_roi = self.current_frame[y:y+h, x:x+w]

            # Display picture with a frame around the face
            self.current_frame = cv2.rectangle(self.current_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

            dim = (48, 48)
            self.face = cv2.resize(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY), dim, interpolation = cv2.INTER_AREA)/255
            
            
        else:
            self.text.insert(END, "No face detected\n")
            self.current_frame = None

    def to_string(self):
        """Convert the result to a string representation."""
        emotion_labels = ['Angry', 'disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        s = max(zip(self.result[0],emotion_labels))[1]
        return s

    def analyse_face(self):
        """Perform emotion analysis on the face and display the result."""
        img_batch = np.expand_dims(self.face, axis=0)
        self.result = self.model.predict(np.asarray(img_batch)) # what is the output?
        print("result = ", self.result)

        # result = self.model(self.face)
        self.text.insert(END, self.text.insert(END, self.to_string() + '\n'))

if __name__ == '__main__':
    root = Tk()
    model = keras.models.load_model('./sequential_model_c.h5')
    app = App(root, model)
    root.mainloop()
    app.release()

    