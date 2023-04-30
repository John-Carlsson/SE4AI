from tkinter import *
import cv2
from PIL import Image, ImageTk

class Camera:
    def __init__(self, width, height):
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
        self.current_frame = None  # Disable capturing
        self.show_video()

    def bad_feedback(self):
        self.current_frame = None  # Disable capturing
        self.show_video()

    def run_analysis(self):
        self.current_frame = self.camera.capture_frame()  # capture a frame
        self.show_video()

        # lucas code


        # Network


    def release(self):
        self.camera.release()

if __name__ == '__main__':
    root = Tk()
    app = App(root)
    root.mainloop()
    app.release()
