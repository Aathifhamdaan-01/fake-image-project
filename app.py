import streamlit as st
from PIL import Image
from transformers import pipeline

@st.cache_resource
def load_detector():

    detector = pipeline(
        "image-classification",
        model="umm-maybe/AI-image-detector"
    )

    return detector

detector = load_detector()

st.title("AI Fake Image Detection")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    result = detector(image)

    st.write(result)