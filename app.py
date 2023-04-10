import streamlit as st
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from skimage.metrics import peak_signal_noise_ratio

st.title('🎈 App Name')

st.write('Hello world!')

import streamlit as st
import numpy as np
import PIL
from io import BytesIO

st.title("Signature Background Remover")

def remove_background(img, thr, softness):
    # Convert the image to RGBA format
    img = img.convert("RGBA")

    # Extract the red, green, and blue channels as NumPy arrays
    r, g, b, a = np.array(img).T

    # Calculate the color intensity of each pixel
    intensity = 0.299 * r + 0.587 * g + 0.114 * b

    # Calculate the alpha channel values based on the color intensity
    alpha = 1 - np.interp(intensity, 
                      [0, max(0, 255 - thr - softness), min(255, 255 - thr + softness), 255], 
                      [0, 0, 1, 1])
    # alpha = np.ones_like(r) * 1.0
    alpha = (alpha * 255).astype(np.uint8)
    # Create a new NumPy array with the modified pixel values
    np_img = np.array([r, g, b, alpha]).T
    # np_img = np.array([r, g, b, alpha]).T
    # Convert the NumPy array to a Pillow image
    # bg = PIL.Image.fromarray(np_img, mode="RGBA")
    # Calculate the intensity histogram of the grayscale image
    
    bg = PIL.Image.fromarray(np_img, mode="RGBA")
    print("ok")
    return bg, alpha



uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    img = PIL.Image.open(uploaded_file)
    st.image(img, caption="Original Image", use_column_width=True)

    # Define the default values for thr_low and thr_high
    default_thr = 80
    default_softness = 10
    
    # Create a slider for thr_low
    thr = st.slider(
        "Select the threshold value", 0, 255, default_thr, 1)
    
    # Create a slider for thr_high
    softness = st.slider(
        "Select the softness value", 0, 255, default_softness, 1)

    st.write("Removing Background...")
    bg, intensity = remove_background(img, thr, softness)
    st.image(bg, caption="Processed Image", use_column_width=True)
    
    img_bytes = BytesIO()
    bg.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    st.download_button(label="Download", data=img_bytes, file_name="background_remover.png")

    img_gray = bg.convert("L")

    # Apply Fourier transform to the image
    img_fft = np.fft.fft2(img_gray)
    
    # Shift zero-frequency component to center of spectrum
    f_shift = np.fft.fftshift(img_fft)
    
    # Compute magnitude spectrum
    mag_spec = np.mean(np.abs(f_shift), axis=0)
    print(mag_spec.shape)
    
    # Convert to logarithmic scale
    # Display the histogram
    fig, ax = plt.subplots()
    plt.plot(np.log(mag_spec))
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")

    st.pyplot(fig)

    
