import os

import cv2
import numpy as np
from mouseinfo import dc

"""
Face detection of face using haarcascade.xml
"""
def detect_face(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier('image_preprocessing/haarcascade_frontalface_alt.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)

    if len(faces) > 0:
        # Get the first detected face So that if multiple persons in frame we just look at one
        (x, y, w, h) = faces[0]

        # Extract the region of interest containing the face
        face_roi = image[y:y + h, x:x + w]

        n = dc.collector()
        e = 1

        # Convert the region to (48 x 48) grayscale and save the face
        dim = (48, 48)
        face = cv2.resize(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY), dim, interpolation=cv2.INTER_AREA) / 255
        n.save_img(face, e)
        return face, True, x, y, x + w, y + h
        cv2.imwrite('./face_test.png', face)
        # print(self.width, self.height)
    else:
        return None, False, 0, 0, 0, 0
"""
Fece detection of face using neuron model from opencv library
@:param image = image from what we want get a image
"""
def detect_face2(image):
     # Define paths
     prototxt_path = os.path.join('image_preprocessing/deploy.prototxt')
     caffemodel_path = os.path.join('image_preprocessing/weights.caffemodel')

     # Read the model
     model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

     cv2.imwrite("imageH.jpg", image)

     image = cv2.imread('imageH.jpg')

     (h, w) = image.shape[:2]
     blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

     model.setInput(blob)
     detections = model.forward()

     # Identify each face
     for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        confidence = detections[0, 0, i, 2]

        # If confidence > 0.5, save it as a separate file
        if (confidence > 0.5):
            frame = image[startY:endY, startX:endX]
            dim = (48, 48)
            face = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dim, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(i) + '_' + "image.jpg", face)
            return face, True, startX, startY, endX, endY
        else:
            return None, False, 0, 0, 0, 0

