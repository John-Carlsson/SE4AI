import os
import wave
import time
import threading
import tkinter as tk
import pyaudio
import numpy as np

from main_ser import add_data_to_queue


audio_array = []



class VoiceRecorder:




    def launch(self):
        self.window = tk.Tk()

        self.window.geometry("1000x650")
        self.window.title("Emotional Recognition through Speech")

        self.phonological_var = tk.StringVar()
        self.linguistic_var = tk.StringVar()


        self.phonological_label = tk.Label(self.window, text="Phonological Prediction",font=("Robot", 18) )
        self.phonological_label.pack()

        self.phonological_prediction = tk.Label(self.window, borderwidth=2, relief="groove", textvariable=self.phonological_var,
                                                width=15, height=3, font=("Robot", 10))
        self.phonological_prediction.pack()


        self.linguistic_label = tk.Label(self.window, text="Linguistic Prediction",  font=("Robot", 18))
        self.linguistic_label.pack()

        self.linguistic_prediction = tk.Label(self.window,borderwidth=2, relief="groove",  textvariable=self.linguistic_var,
                                              width=15, height=3, font=("Robot", 10))
        self.linguistic_prediction.pack()

        self.button = tk.Button(text="Record", font=("Robot", 80, "bold"),
                                command=self.record_signal)
        self.button.pack()

        self.time_label = tk.Label(text="00:00", pady=5)
        self.time_label.pack()

        self.feedback_label = tk.Label(self.window, text="Was the predicition RIGHT or WRONG", font=("Robot", 18))
        self.feedback_label.pack()

        self.button_right = tk.Button(text="RIGHT", font=("Robot", 20, "bold"), fg="green", pady=5)
        self.button_right.pack()

        self.button_left = tk.Button(text="WRONG", font=("Robot", 20, "bold"), fg="red", pady=5)
        self.button_left.pack()





        self.recording = False
        self.window.mainloop()



    def record_signal(self):
        if self.recording:
            self.recording = False
            self.button.config(fg="black")
        else:
            self.recording = True
            self.button.config(fg="red")
            print("Record started")
            threading.Thread(target=self.record).start()



    def record(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100,
                            input=True, frames_per_buffer=1024)


        start = time.time()

        while self.recording:
            #read the data from the stream
            audio_data = stream.read(1024)
            audio_array.append(np.frombuffer(audio_data, dtype=np.float32))



            passed = time.time() - start

            secs = passed % 60
            mins = passed // 60
            self.time_label.config(text=f"{int(mins):02d}:{int(secs):02d}")

            if passed >= 5:
                self.recording = False
                self.button.config(fg="black")
                #only for testing
                #self.publish_emotion_label("hallo", "moin")

        print("Record stopped")
        stream.stop_stream()
        stream.close()
        audio.terminate()

        audio_array_converted = np.concatenate(audio_array)

        add_data_to_queue(audio_array_converted)




    def publish_emotion_label(self, prediction_1, prediction_2):

        self.phonological_var.set(prediction_1)
        self.linguistic_var.set(prediction_2)




if __name__ == "__main__":
    vc = VoiceRecorder()
    vc.launch()