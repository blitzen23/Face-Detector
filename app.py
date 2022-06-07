import streamlit as st
import cv2
import os
import sys
import numpy as np

from PIL import Image

cascPath = "./haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascPath)

st.title('Face Detector')

videoBtn = st.button('Video')

if videoBtn:   
    stopBtn = st.button('Stop')
    video = st.empty()
    facesFound = st.empty()
    webcam = cv2.VideoCapture(0)
    while True:
        _, image = webcam.read()

        faces = faceCascade.detectMultiScale(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(30, 30),
        )
        
        for face in faces:
            cv2.rectangle(image, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (255, 0, 0), 3)

        video.image(image)
        facesFound.text("%d faces found or detected!" %(len(faces)))

    webcam.release()
    cv2.destroyAllWindows()

uploadedFile = st.file_uploader('Choose an image', type=['png', 'jpg', 'jpeg', 'webp'])
if uploadedFile is not None:
    image = Image.open(uploadedFile)
    image = image.save('img.jpg')
    image = cv2.imread("img.jpg")

    faces = faceCascade.detectMultiScale(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(30, 30),
    )

    for face in faces:
        cv2.rectangle(image, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (255, 0, 0), 3)
        print(face[0], face[1], face[2], face[3])

    st.image(image)
    st.write("%d faces found or detected!" %(len(faces)))