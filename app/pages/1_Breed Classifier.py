import streamlit as st
from PIL import Image
import tempfile
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.breed_classification.predict import predict_breed


st.title("🐄 Breed Classification")

st.write("Upload a cow image to identify its breed.")


uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict Breed"):

        file_extension = uploaded_file.name.split(".")[-1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name

        breed, confidence = predict_breed(temp_path)

        st.success(f"Predicted Breed: {breed}")
        st.info(f"Confidence: {confidence:.2f}%")