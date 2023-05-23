# imports
import os
import sys
sys.path.append(os.path.join(os.path.realpath(__file__), "preprocessing_ser.py"))
from preprocessing_ser import calculate_spectrograms, pad_with_zeros
sys.path.append(os.path.join(os.path.realpath(__file__), "CRNN_LSTM_model.py"))
from CRNN_LSTM_model import CRNN_LSTM
sys.path.append(os.path.join(os.path.realpath(__file__), "Semantic_approach.py"))
from Semantic_approach import Semantic_Approach




data_queue = []



# stop button
def add_data_to_queue(data):
    data_queue.append(data)





def pipeline():
    while True:
        if data_queue:
            data_sample = data_queue.pop(0)

            # Preprocessing for phonological info
            if len(data_sample) < 18000: # TODO
                data_sample = pad_with_zeros(data_sample)
            spec = calculate_spectrograms(data_sample)
        


        # Prediction based on phonological info
        probs, prediction_1 = model.predict(spec, single_sample=True)


        # Prediction based on linguistic info
        prediction_2 = sem_appr.speech_to_emotion(data_sample)



        # User interface
        # ui.publish_emotion_label([prediction_1, prediction_2])

        predictions_dic = {
            "phonological info": [prediction_1, probs],
            "linguistic info": prediction_2
        }






if __name__ == "__main__":
    model = CRNN_LSTM(model_name="trial_data_model")
    sem_appr = Semantic_Approach()

    #ui = UserInterface()
    #ui.launch()
    # start user interface
        # interface stops recording after 5 sec

    pipeline()

    


        
        




