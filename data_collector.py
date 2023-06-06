import csv
import cv2
import keras
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


class collector:
    def __init__(self, src = './ImageColection/collection.csv'):
        self.src = src

    def create_new_file(self, dirc = './ImageColection/collection.csv'):
        with open(dirc, 'w', newline='') as file:
            writer = csv.writer(file)
            field = ["emotion", "pixels", "Usage"]
            writer.writerow(field)


    def reset_default(self):
        self.create_new_file()


    def string_builder(self, flattend_list):
        """
        Parameters
        ----------
        flattend_list : list
            List of pixels to be converted to a string
        """
        strg = ''
        for p in flattend_list:
            strg += str(round(p*255, 3)) + ' '
        return strg


    def save_img(self, image, e, datatype = "PrivateTest"):
        """
        Parameters
        ----------
        image : ndarray
            A (48x48) array of a grayscaled image captured with the cv2 library
        e : int
            The correct emotion (Value [1:7])
        datatype : string, optional
            Either Training, PublicTest or PrivateTest, default is the later.
        """
        with open(self.src, 'a') as file:
            writer = csv.writer(file)
            df = [str(e), self.string_builder(list(np.concatenate(image).flat)), datatype]
            writer.writerow(df)

