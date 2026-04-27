import streamlit as st
import requests

st.set_page_config(page_title="GTZAN Audio Classifier", layout="centered")

st.title("GTZAN Audio Classification App")
st.write("Upload an audio file and get prediction from the model")

API_URL = "http://127.0.0.1:8000/predict/"

uploaded_file = st.file_uploader(
    "Upload audio file",
    type=["wav", "mp3", "flac"]
)

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("Predict"):
        files = {
            "file": (uploaded_file.name, uploaded_file, uploaded_file.type)
        }

        try:
            with st.spinner("Processing..."):
                response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                result = response.json()

                st.success(f"Model thinks it's: {result['Class']}")
            else:
                st.error(f"Error: {response.text}")

        except Exception as e:
            st.error(f"Connection error: {e}")
