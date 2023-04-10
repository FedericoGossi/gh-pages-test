import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import PIL
from io import BytesIO
from tqdm.auto import tqdm

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

    # Convert to logarithmic scale
    # Display the histogram
    img = img.convert("RGBA")
    r, g, b, a = np.array(img).T
    intensity = 0.299 * r + 0.587 * g + 0.114 * b
    non_zeros = []
    xs = np.arange(0, 255)
    for thr in tqdm(xs):
      alpha = 1 - np.interp(intensity, 
                        [0, max(0, 255 - thr - softness), min(255, 255 - thr + softness), 255], 
                        [0, 0, 1, 1])
      non_zeros.append((alpha!=0).sum())

    der = lambda arr: [arr[i] - arr[i-1] for i in range(1, len(arr))]
    non_zeros_2nd_der = der(non_zeros)
    fig, ax = plt.subplots()
    # xss = list(range(1, len(non_zeros_2nd_der)+1))
    xss = list(range(len(non_zeros)))
    plt.plot(xss, non_zeros)
    plt.ylabel("Non-transparent pixel count")
    plt.xlabel("Threshold")
    
    
    argmax = np.argmin(non_zeros_2nd_der)
    # q = min(non_zeros_2nd_der) * 0.01
    q = - non_zeros[0] / 255 * 0.1
    # plt.plot(xss, [q*x + non_zeros[0]*0.1 for x in xss])
    peak_vals = [i if non_zeros_2nd_der[i] < q else -1 for i in range(len(non_zeros_2nd_der))]
    # pos = max(peak_vals)+1
    pos = 255
    for i in range(np.argmin(non_zeros_2nd_der), len(peak_vals)):
      if peak_vals[i] == -1 and peak_vals[i-1] != -1:
        pos = i
        break
    plt.axvline(x=pos, ls='--', color='red')
    # Set the x-axis ticks every 10 units
    ticklocations = np.arange(0, 256, 10)
    ax.set_xticks(ticklocations, minor=True)
    ax.set_xticks(np.arange(0, 256, 50), minor=False)

    # Set the x-axis tick labels to be shown every 50 units
    ax.set_xticklabels(np.arange(0,256,50))
    # st.write("q:", q)
    
    st.pyplot(fig)
    st.write("Suggested threshold:", pos)
    
    
