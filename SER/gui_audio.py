import os
import wave
import time
import threading
import tkinter as tk
import pyaudio
import numpy as np


audio_array = []

shared_variable = ""

phonological_prediction = None

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


        self.button = tk.Button(text="Record", font=("Robot", 80, "bold"),
                                command=self.record_signal)
        self.button.pack()

        self.time_label = tk.Label(text="00:00", pady=5)
        self.time_label.pack()

        self.feedback_label = tk.Label(self.window, text="Please select the used emotion", font=("Robot", 18))
        self.feedback_label.pack()

        self.button_frame = tk.Frame(self.window)
        self.button_frame.pack()

        self.button_angry = tk.Button(self.button_frame, text="ANGRY", font=("Robot", 20, "bold"), fg="black", pady=5)
        self.button_angry.grid(row=0, column=0, padx=5, pady=5)

        self.button_disgusting = tk.Button(self.button_frame, text="DISGUSTING", font=("Robot", 20, "bold"), fg="black",
                                           pady=5)
        self.button_disgusting.grid(row=0, column=1, padx=5, pady=5)

        self.button_fear = tk.Button(self.button_frame, text="FEAR", font=("Robot", 20, "bold"), fg="black", pady=5)
        self.button_fear.grid(row=0, column=2, padx=5, pady=5)

        self.button_happy = tk.Button(self.button_frame, text="HAPPY", font=("Robot", 20, "bold"), fg="black", pady=5)
        self.button_happy.grid(row=1, column=0, padx=5, pady=5)

        self.button_neutral = tk.Button(self.button_frame, text="NEUTRAL", font=("Robot", 20, "bold"), fg="black",                                       pady=5)
        self.button_neutral.grid(row=1, column=1, padx=5, pady=5)

        self.button_sad = tk.Button(self.button_frame, text="SAD", font=("Robot", 20, "bold"), fg="black", pady=5)
        self.button_sad.grid(row=1, column=2, padx=5, pady=5)

        #6 button mit den einzelnen emotionen

        #falls thread nicht klappt, einfach die recorded audio in die method, falls data not null

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
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000,
                            input=True, frames_per_buffer=512)


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

        from main_ser import add_data_to_queue
        add_data_to_queue(self, audio_array_converted)

    def update_label_text(self, shared_variable):

        new_text = shared_variable
        self.phonological_var.set(new_text)



def publish_emotion_label(self, prediction_1, prediction_2):


        global shared_variable
        shared_variable = prediction_1[0]
        self.update_label_text(shared_variable)





if __name__ == "__main__":
    vc = VoiceRecorder()
    vc.launch()
