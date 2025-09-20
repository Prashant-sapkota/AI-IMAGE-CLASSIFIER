import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image


def load_model():
    model = MobileNetV2(weights='imagenet')
    return model

def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img


def classify_image(model, image):
    try:
        preproced_image = preprocess_image(image)
        predections = model.predict(preproced_image)
        decoded_predictions = decode_predictions(predections, top=3)[0]
        
        return decoded_predictions
    except Exception as e:
        st.error(f"Error during classification: {e}")
        return None
    

def main():
    st.set_page_config(page_title="AI Image Classifier", page_icon="🖼️" ,layout="centered")
    st.title("AI IMAGE CLASSIFIER")
    st.write("Upload an image to classify and see the top predictions.")
        
    
    @st.cache_data
    def load_cached_model():
        return load_model()
    
    model = load_cached_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","webp", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = st.image(image, caption='Uploaded Image', use_container_width=True)
        
        btn = st.button('Classify Image')

        if btn:
            with st.spinner('Analysing.. . ..'):
                 image = Image.open(uploaded_file)
                 predictions = classify_image(model, image)
                
                 if predictions:
                    st.subheader('predections Complete!')
                    for _, label, score in predictions:
                        st.write(f"**{label}**: {score:.2%}")


if __name__ == "__main__":
    main()

                