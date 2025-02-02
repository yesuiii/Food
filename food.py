import streamlit as st
import fastai
from fastai.vision.all import *
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    learn = load_learner('mongolian_food_classifier.pkl')
    return learn

def main():
    st.title("Mongolian Food Classifier")
    st.write("Upload an image")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        learn = load_model()

        pred, pred_idx, probs = learn.predict(image)

        # Display results
        st.write(f"### Prediction: {pred}")
        st.write(f"### Confidence: {probs[pred_idx]:.4f}")

if __name__ == "__main__":
    main()
