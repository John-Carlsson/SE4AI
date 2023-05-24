import os
import wave
import time
import threading
import tkinter as tk
import pyaudio
import np

#from main_ser import add_data_to_queue


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


        self.time_label = tk.Label(text="00:00")
        self.time_label.pack()

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

        #add_data_to_queue(audio_array_converted)




    def publish_emotion_label(self, prediction_1, prediction_2):

        self.phonological_var.set(prediction_1)
        self.linguistic_var.set(prediction_2)


        #exists = True
        #i = 1

        #while exists:
        #   if os.path.exists(f"recording{i}.wav"):
        #        i += 1;
        #    else:
        #        exists = False

        #sound_file = wave.open(f"recording{i}.wav", "wb")
        #sound_file.setnchannels(1)
        #sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        #sound_file.setframerate(44100)
        #sound_file.writeframes(b"".join(frames))
        #sound_file.close()



if __name__ == "__main__":
    vc = VoiceRecorder()
    vc.launch()