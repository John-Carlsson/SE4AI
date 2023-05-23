import os
import wave
import time
import threading
import tkinter as tk
import pyaudio

class VoiceRecorder:

    def __init__(self):
        self.window = tk.Tk()

        #generate base window

        self.output_console = tk.Text(self.window)
        self.output_console.pack()
        self.window.resizable(False, False)

        #add the record button
        self.button = tk.Button(text="Record", font=("Robot", 80, "bold"),
                                command=self.click_handler)
        self.button.pack()
        self.label_time = tk.Label(text="00:00")
        self.label_time.config(font=("Robot", 18, "bold"))
        self.label_time.pack()

        self.label_phonological = tk.Label(text="Phonological Prediciton")
        self.label_phonological.config(font=("Robot", 18, "bold"))
        self.label_phonological.pack()

        self.label_linguistic = tk.Label(text="Linguistic Prediciton")
        self.label_linguistic.config(font=("Robot", 18, "bold"))
        self.label_linguistic.pack()




        self.recording = False
        self.window.mainloop()

    def click_handler(self):
        if self.recording:
            self.recording = False
            self.button.config(fg="black")
        else:
            self.recording = True
            self.button.config(fg="red")
            threading.Thread(target=self.record).start()

    def record(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100,
                            input=True, frames_per_buffer=1024)

        frames = []

        start = time.time()

        while self.recording:
            data = stream.read(1024)
            frames.append(data)

            passed = time.time() - start

            secs = passed % 60
            mins = passed // 60
            self.label.config(text=f"{int(mins):02d}:{int(secs):02d}")

        stream.stop_stream()
        stream.close()
        audio.terminate()

        exists = True
        i = 1

        while exists:
            if os.path.exists(f"recording{i}.wav"):
                i += 1;
            else:
                exists = False

        sound_file = wave.open(f"recording{i}.wav", "wb")
        sound_file.setnchannels(1)
        sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(44100)
        sound_file.writeframes(b"".join(frames))
        sound_file.close()


VoiceRecorder()