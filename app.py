import streamlit as st
import numpy as np
import cv2

def remove_background(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.merge((thresh, thresh, thresh))
    img = cv2.bitwise_and(img, mask)
    return img

def create_mask(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold the grayscale image to create a binary mask
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    # Convert the binary mask to a RGBA mask with variable transparency
    mask_rgba = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGBA)
    mask_rgba[:, :, 3] = np.where(mask == 255, 255, np.uint8(0.5*(255-mask)))
    return mask_rgba

st.title("Signature Background Remover")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    st.image(img, caption="Original Image", use_column_width=True)

    st.write("Removing Background...")

    img = remove_background(img)

    st.image(img, caption="Processed Image", use_column_width=True)
