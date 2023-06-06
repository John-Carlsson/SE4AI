from tkinter import *
import cv2
import keras
import numpy as np
from PIL import Image, ImageTk
import platform
import data_collector as dc
from functools import partial
from keras.optimizers import Adam
import os
import wave
import time
import threading
import pyaudio
import pandas as pd
import time

audio_array = []
recorded_dataframes = []

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

        self.good_button = Button(self.master, text='Correct', font=('arial', 25), fg='green', command=self.good_feedback, state= DISABLED)
        self.bad_button = Button(self.master, text='False', font=('arial', 25), fg='red', command=self.bad_feedback, state= DISABLED)
        self.capture_button = Button(self.master, text='Capture & Record', font=('arial', 25), fg='black', command=self.both_methods)
        self.time_label = Label(text="00:00", pady=5)
        self.recording = False
        self.time_label.pack()
        self.capture_button.pack(side='bottom')
        self.bad_button.pack(side='bottom')
        self.good_button.pack(side='bottom')

        # emotion label: 'Angry', 'disgust', 'Fear', 'Happy', 'Sad', 'Neutral'
        self.happy_button = Button(self.master, text='Happy', font=('arial', 25), fg='black', command=lambda: self.bad_real_emotion(np.array([[0, 0, 0, 1, 0, 0, 0]])))
        self.sad_button = Button(self.master, text='Sad', font=('arial', 25), fg='black', command=lambda: self.bad_real_emotion(np.array([[0, 0, 0, 0, 1, 0, 0]])))
        self.neutral_button = Button(self.master, text='Neutral', font=('arial', 25), fg='black', command=lambda: self.bad_real_emotion(np.array([[0, 0, 0, 0, 0, 0, 1]])))
        self.fear_button = Button(self.master, text='Fear', font=('arial', 25), fg='black', command=lambda: self.bad_real_emotion(np.array([[0, 0, 1, 0, 0, 0, 0]])))
        self.angry_button = Button(self.master, text='Angry', font=('arial', 25), fg='black', command=lambda: self.bad_real_emotion(np.array([[1, 0, 0, 0, 0, 0, 0]])))
        # self.suprise_button = Button(self.master, text='Surprise', font=('arial', 25), fg='black', command=lambda: self.bad_real_emotion(np.array([[0, 0, 0, 0, 0, 1, 0]])))
        self.disgust_button = Button(self.master, text='Disgust', font=('arial', 25), fg='black', command=lambda: self.bad_real_emotion(np.array([[0, 1, 0, 0, 0, 0, 0]])))

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
            self.update_model_with_feedback(np.array([[0, 0, 0, 0, 0, 0, 0]]))
            self.capture_button.config(state=NORMAL)
            self.good_button.config(state=DISABLED)
            self.bad_button.config(state=DISABLED)

    def bad_feedback(self):
        """Handle the 'False' button click event."""
        if self.current_frame is not None:
            
            self.happy_button.pack(side='left')
            self.sad_button.pack(side='left')
            self.neutral_button.pack(side='left')
            self.fear_button.pack(side='left')
            self.angry_button.pack(side='left')
            self.disgust_button.pack(side='left')


    def bad_real_emotion(self, true_emotion):
        """Handle the 'False' button click event and takes feedback for real emotion."""
        if self.current_frame is not None:
            
            #how does feedback function work?
            self.result = true_emotion

            self.happy_button.pack_forget()
            self.sad_button.pack_forget()
            self.neutral_button.pack_forget()
            self.fear_button.pack_forget()
            self.angry_button.pack_forget()
            self.disgust_button.pack_forget()


            self.current_frame = None  # Disable stillframe
            self.show_video()
            self.capture_button.config(state=NORMAL)
            self.good_button.config(state=DISABLED)
            self.bad_button.config(state=DISABLED)
            self.update_model_with_feedback(true_emotion)

            
    def update_model_with_feedback(self, true_emotion):
        print("update model")
        """Update the model with the provided feedback."""
        #if self.feedback is not None:
        img_batch = np.expand_dims(self.face, axis=0)
        
        # true_emotion == 0 array represents correct prediction -> correct prediction gets saved
        if np.array_equal(true_emotion, np.array([[0, 0, 0, 0, 0, 0, 0]])):
            idx = np.where(max(self.result[0]) == self.result[0])
            self.result[0] = [0 for i in range(len(self.result[0]))]
            self.result[0][idx] = 1

            feedback_data = self.result
            print("feedback data: " + str(feedback_data))
        # if prediction is wrong and true emotion was given
        else:
            feedback_data = true_emotion
            print("feedback data false: " + str(feedback_data))

        optimizer = Adam(learning_rate=0.001)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.model.fit(img_batch, feedback_data, epochs=1)

        self.feedback = None

    # Capture button
    # TO-DO: input audio function
    def run_analysis(self):

        print("moin")
        """Perform face analysis on the captured frame."""
        self.current_frame = self.camera.capture_frame()  # capture a frame
        # call function to record audio
        self.detect_face()
        self.show_video()
        if self.current_frame is not None:
            self.analyse_face()


    def both_methods(self):
        self.run_analysis()
        self.record_signal()


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
        emotion_labels = ['Angry', 'disgust', 'Fear', 'Happy', 'Sad', 'Neutral']
        s = max(zip(self.result[0],emotion_labels))[1]
        return s

    def analyse_face(self):
        """Perform emotion analysis on the face and display the result."""
        img_batch = np.expand_dims(self.face, axis=0)
        self.result = self.model.predict(np.asarray(img_batch)) # what is the output?
        print("result = ", self.result)

        # result = self.model(self.face)
        time.sleep(1)
        self.text.insert(END, self.to_string() + '\n')

        return self.result


    def record_signal(self):
        if self.recording:
            self.recording = False
            self.capture_button.config(fg="black")
        else:
            self.recording = True
            self.capture_button.config(fg="red")
            print("Record started")
            threading.Thread(target=self.record).start()

    def record(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000,
                            input=True, frames_per_buffer=512)

        start = time.time()

        while self.recording:
            # read the data from the stream
            audio_data = stream.read(1024, exception_on_overflow = False)
            audio_array.append(np.frombuffer(audio_data, dtype=np.float32))

            passed = time.time() - start

            secs = passed % 60
            mins = passed // 60
            self.time_label.config(text=f"{int(mins):02d}:{int(secs):02d}")
            self.capture_button.config(state=DISABLED)

            if passed >= 5:
                self.recording = False
                self.capture_button.config(fg="black")
                # only for testing
                # self.publish_emotion_label("hallo", "moin")
                self.bad_button.config(state=NORMAL)
                self.good_button.config(state=NORMAL)

        print("Record stopped")
        stream.stop_stream()
        stream.close()
        audio.terminate()

        audio_array_converted = np.concatenate(audio_array)
        #self.set_audio_for_feedback(audio_array_converted)

        from main import add_data_to_queue
        add_data_to_queue(self, audio_array_converted)

        #create_dataframe_audio_and_feedback(self)

        def update_label_text(self, shared_variable):
            new_text = shared_variable
            self.phonological_var.set(new_text)

        def set_user_feedback(self, emotion):
            self.user_feedback_emotion = emotion

        def get_user_feedback(self):
            return self.user_feedback_emotion

        def set_audio_for_feedback(self, recorded_audio):
            self.recorded_audio = recorded_audio

        def get_audio_for_feedback(self):
            return self.recorded_audio

def create_dataframe_audio_and_feedback(self):

        emotion = self.get_user_feedback()
        audio = self.get_audio_for_feedback()

        data = {'Emotion': [emotion], 'Audio Array': [audio]}
        df = pd.DataFrame(data)
        recorded_dataframes.append(df)

        print(recorded_dataframes)


def publish_emotion_label(self, prediction_1):
    global shared_variable
    shared_variable = prediction_1[0]
    print(shared_variable)

    return shared_variable

    #self.update_label_text(shared_variable)


def sending_results(self):

    result_face = self.analyse_face()
    result_voice = self.publish_emotion_label()



    from main import combine_results
    combine_results(result_face, result_voice)

# if __name__ == '__main__':
#     root = Tk()
#     model = keras.models.load_model('./sequential_d5_model_c.h5')
#     app = App(root, model)
#     root.mainloop()
#     app.release()
