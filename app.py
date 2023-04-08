import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title='Image Inverter')

st.title('Image Inverter')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Original Image', use_column_width=True)
    image = Image.open(uploaded_file)
    inverted_image = ImageOps.invert(image)
    st.image(inverted_image, caption='Inverted Image', use_column_width=True)
