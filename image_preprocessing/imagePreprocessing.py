"""
How to import OpenCV to PyCharm
1) Open File>Settings>Project
2) Select your current project and click the Python Interpreter
3) click small + to add library
4) Install 'opencv-python'
"""
#the opecCV library
import cv2

# read the input image
img = cv2.imread('picture1.jpg')

# convert to grayscale of each frames
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# read the haarcascade to detect the faces in an image
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# detects faces in the input image
faces = face_cascade.detectMultiScale(gray, 1.3, 4)
print('Number of detected faces:', len(faces))

# loop over all detected faces
if len(faces) > 0:
    for i, (x, y, w, h) in enumerate(faces):
        # To draw a rectangle in a face
        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        face = img[y:y + h, x:x + w]
        gray_image = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Cropped Face", gray_image)
        cv2.imwrite(f'face{i}.jpg', gray_image)
        print(f"face{i}.jpg is saved")

# display the image with detected faces
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()