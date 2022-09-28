import streamlit as st


## image
from PIL import Image
image = Image.open('pythoncoding.jpg')
st.image(image, caption='Chilling with Python')


## video
video_file = open('whale_vid.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)