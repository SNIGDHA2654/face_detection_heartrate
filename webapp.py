import streamlit as st
import os
import numpy as np 
from PIL import Image
import cv2
import get_pulse

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_face(our_image):
    img = np.array(our_image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5)
    for x,y,w,h in face:
        cv2.rectangle(img, (x , y), (x+w, y+h), (0,255,0), 3)
    
    return img

def open_cam():
    video = cv2.VideoCapture(0)

    while True:
        check, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5)
        for x,y,w,h in face:
            img = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
        cv2.imshow('gotcha', frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

    video.release()

def main():
    #The Interface
    st.title("Image Recognition")

    html_temp = """
    <body style="background-color : blue;">
    <div style="background-color : teal; pading:10px">
    <h2 style="color:white; text-align:center;"> Face </h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if image_file is not None:
        our_image = Image.open(image_file)
        st.text('Original Image')
        st.image(our_image)

    if st.button('Recognise'):
        result_img = detect_face(our_image)
        
        st.image(result_img)

    if st.button('Camera'):
        open_cam()

    if st.button('Check Hear Rate'):
        os.system('python get_pulse.py')

if __name__ == '__main__':
    main()
