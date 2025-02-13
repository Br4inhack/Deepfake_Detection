import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from mtcnn import MTCNN

# Set page configuration
st.set_page_config(page_title="Deepfake Detection", layout="wide")

# Function to load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('Streamlit_Application_for_DeepFake.h5')
    return model

# Function to check if a given video or image contains deepfake
def is_deepfake(file_path, model):
    detector = MTCNN()
    cap = cv2.VideoCapture(file_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        faces = detector.detect_faces(frame)
        for face in faces:
            x, y, width, height = face['box']
            face_img = frame[y:y+height, x:x+width]
            face_img = cv2.resize(face_img, (224, 224))
            face_img = np.expand_dims(face_img, axis=0)
            prediction = model.predict(face_img)
            if prediction > 0.5:
                cap.release()
                return True
    cap.release()
    return False

# Load the model
model = load_model()

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #1f1f1f;
    }
    .title {
        font-size: 2em;
        font-weight: bold;
    }
    .subtitle {
        font-size: 1.2em;
        color: gray;
    }
    .button {
        background-color: blue;
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 1em;
        cursor: pointer;
    }
    .linkedin-box {
        border: 2px solid #ddd;
        padding: 20px;
        margin-top: 20px;
    }
    .linkedin-profile {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    .linkedin-logo {
        width: 20px;
        height: 20px;
        margin-right: 10px;
    }
    .result{
        display: flex;
        text-items: center;
        font-size: 3em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Page title
st.markdown('<div class="title">Deepfake Detection</div>', unsafe_allow_html=True)

# Hashtag subtitle
st.markdown('<div class="subtitle">Abki Baar Deepfakes Par Varr</div>', unsafe_allow_html=True)

# Create two columns for the boxes
col1, col2 = st.columns(2)

# Add boxes to the columns
with col1:
    uploaded_file = st.file_uploader("Upload Video or Image", type=["jpg", "jpeg", "png", "mp4", "avi"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        if uploaded_file.type in ["image/jpeg", "image/png"]:
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = MTCNN().detect_faces(img_rgb)
            with st.container():
                st.markdown('<div class="box">', unsafe_allow_html=True)
                st.image(img_rgb, channels="RGB")
                st.markdown('</div>', unsafe_allow_html=True)
            if faces:
                x, y, width, height = faces[0]['box']
                face_img = img_rgb[y:y+height, x:x+width]
                face_img = cv2.resize(face_img, (224, 224))
                face_img = np.expand_dims(face_img, axis=0)
                prediction = model.predict(face_img)
                if prediction > 0.5:
                    st.markdown('<div class="result">This is a deepfake.',unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result">This is not a deepfake.',unsafe_allow_html=True)
            else:
                st.markdown('<div class="result">No face detected in the image.',unsafe_allow_html=True)
        elif uploaded_file.type in ["video/mp4", "video/avi"]:
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.video("temp_video.mp4")
            result = is_deepfake("temp_video.mp4", model)
            if result:
                st.markdown('<div class="result">This is a deepfake.',unsafe_allow_html=True)
            else:
                st.markdown('<div class="result">##This is not a deepfake.',unsafe_allow_html=True)

with col2:
    st.markdown('### Understanding Deepfakes', unsafe_allow_html=True)
    st.write("""
        Deepfakes are synthetic media in which a person in an existing image or video is replaced with someone else's likeness.
        They are created using deep learning techniques and can be used for both benign and malicious purposes.

        **Why are deepfakes concerning?**
        - They can spread misinformation and fake news.
        - They can be used for identity theft and fraud.
        - They can damage reputations and invade privacy.

        **How to detect deepfakes?**
        - Look for unnatural facial movements or expressions.
        - Check for inconsistencies in lighting and shadows.
        - Use specialized software and AI models to analyze media files.

        By contributing to this project, you help in the fight against malicious use of deepfakes. Thank you for your support!
    """)
    if st.button("Contribute to the project"):
        st.markdown("[Here's your Link to the Github Repo]()")
        st.write("Thankyou for your contiribution!!")
# Add a placeholder to balance the length of col1 and col2
st.markdown('<div style="height: 100px;"></div>', unsafe_allow_html=True)

# Add LinkedIn section
st.markdown('## Connect with Us on LinkedIn')

col1, col2 = st.columns(2)
linkedin_logo_url = "https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png"

for i, profile in enumerate(linkedin_profiles):
    if i % 2 == 0:
        with col1:
            st.markdown(
                f"""
                <div class="linkedin-profile">
                    <img src="{linkedin_logo_url}" class="linkedin-logo">
                    <a href="{profile['url']}">{profile['name']}</a>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        with col2:
            st.markdown(
                f"""
                <div class="linkedin-profile">
                    <img src="{linkedin_logo_url}" class="linkedin-logo">
                    <a href="{profile['url']}">{profile['name']}</a>
                </div>
                """,
                unsafe_allow_html=True
            )
