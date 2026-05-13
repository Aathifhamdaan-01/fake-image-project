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
    "Choose an image...",
    type=["png", "jpg", "jpeg", "jfif"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    result = detector(image)

    prediction = result[0]

    label = prediction["label"]
    score = prediction["score"]

    if label == "artificial":
        st.error(f"AI Generated Image Detected ({score:.2%} confidence)")
    else:
        st.success(f"Real Human Image ({score:.2%} confidence)")