import numpy as np

default_emotion_mapping = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Neutral']
    

def combine_results(fer_result: np.array, ser_result: np.array, weight_fer=0.5, emotion_mapping=default_emotion_mapping):
    """ Combines the results of the two individual models, applying a weight on each of the result.
    
    Parameters
    ----------
    fer_result : np.array
        The probabilities predicted by the FER model
    ser_result : np.array
        The probabilities predicted by the SER model
    weight_fer : float
        The weighting factor by which the FER probabilities are multiplied. The SER probabilities are multiplied by the inverse.
    emotion_mapping : list
        The emotion labels in a specified order
    
    Returns
    -------
    combined_result : string
        a string representing the most likely emotion
    
    """

    weighted_fer_result = np.multiply(fer_result, weight_fer)
    weighted_ser_result = np.multiply(ser_result, (1 - weight_fer))

    combined_result = np.add(weighted_fer_result, weighted_ser_result)
    most_likely_index = np.argmax(combined_result, axis=0)
    most_likely_label = emotion_mapping[most_likely_index]

    return most_likely_label






if __name__ == "__main__":
    probs_fer = [0.25, 0.15, 0.1, 0.15, 0.3, 0.05]
    probs_ser = [0.25, 0.15, 0.1, 0.15, 0.05, 0.3]

    print(combine_results(probs_fer, probs_ser, weight_fer=0.7))

