from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pyarrow import lib 
import pyarrow.lib as _lib
from datasets import load_dataset
from transformers import pipeline
import os
import pickle

default_path = "Models"
default_name = "Linguistic"


class Semantic_Approach:

    def __init__(self, path=default_path, name=default_name):
        self.path = path
        self.name = name

        if os.path.isfile(os.path.join(self.path, self.name) + '.pkl'):
            pkl_file = open(os.path.join(self.path, self.name) + '.pkl', 'rb')          
            self.asr_processor, self.asr_model, self.em_model = pickle.load(pkl_file)
            pkl_file.close()
            print("Loaded models from existing file")
        else:
            self.asr_processor = WhisperProcessor.from_pretrained("openai/whisper-large")
            self.asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
            self.asr_model.config.forced_decoder_ids = None
            self.em_model = pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)
            print("Downloaded pre-trained models")
        


    def speech_to_text(self, audio):
        """ Sampling/frame rate of audio input must be 16kHz.
        """
        input_features = self.asr_processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        # generate token ids
        predicted_ids = self.asr_model.generate(input_features)
        # decode token ids to text
        transcription = self.asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription
    


    def text_to_emotion(self, text):
        prediction = self.em_model(text)
        max_em = prediction[0][0]
        for i in range(len(prediction[0])):
            if prediction[0][i]['score'] > max_em['score']:
                max_em = prediction[0][i]
        return max_em['label']



    def speech_to_emotion(self, audio):
        """ Sampling/frame rate of audio input must be 16kHz.
        """
        return self.text_to_emotion(self.speech_to_text(audio))



    def store_models(self):
        output = open(os.path.join(self.path, self.name) + '.pkl', 'wb')

        # Pickle dictionary using protocol 0.
        pickle.dump([self.asr_processor, self.asr_model, self.em_model], output)

        output.close()




if __name__ == "__main__":
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sample = ds[0]["audio"]
    sem_appr = Semantic_Approach()
    transcription = sem_appr.speech_to_text(sample["array"])
    print(transcription)
    prediction = sem_appr.text_to_emotion(transcription)
    print(prediction)
    sem_appr.store_models()