import streamlit as st
from PIL import Image
from transformers import pipeline

# PAGE CONFIG
st.set_page_config(
    page_title="AI Fake Image Detector",
    page_icon="🤖",
    layout="centered"
)

# CUSTOM HTML + CSS
st.markdown("""
<style>

body {
    background-color: #0E1117;
}

.title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #00FFAA;
    margin-top: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #CCCCCC;
    margin-bottom: 30px;
}

.stButton>button {
    background-color: #00FFAA;
    color: black;
    font-size: 18px;
    border-radius: 10px;
    padding: 10px 20px;
}

</style>

<div class="title">
🤖 AI Fake Image Detection System
</div>

<div class="subtitle">
Upload an image to check whether it is AI-generated or real.
</div>

""", unsafe_allow_html=True)

# LOAD MODEL
@st.cache_resource
def load_detector():

    detector = pipeline(
        "image-classification",
        model="umm-maybe/AI-image-detector"
    )

    return detector

detector = load_detector()

# SIDEBAR
st.sidebar.title("About")
st.sidebar.info(
    "This AI model detects whether an uploaded image is AI-generated or a real human image."
)

# FILE UPLOAD
uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=["png", "jpg", "jpeg", "jfif"]
)

# IMAGE PROCESSING
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

    st.write("Image Size:", image.size)

    with st.spinner("🔍 Analyzing image..."):

        result = detector(image)

    prediction = result[0]

    label = prediction["label"]
    score = prediction["score"]

    # PROGRESS BAR
    st.progress(int(score * 100))

    # RESULT
    if label == "artificial":

        st.error(
            f"🚨 AI Generated Image Detected ({score:.2%} confidence)"
        )

    else:

        st.success(
            f"✅ Real Human Image ({score:.2%} confidence)"
        )