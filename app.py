import streamlit as st
import pickle
import pandas as pd
from PIL import Image

# Load saved model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Page config
st.set_page_config(page_title="Fruit Taste Classifier",
                   page_icon="🍓",
                   layout="centered")

# Header
st.markdown("<h1 style='text-align:center; color:#FF6600;'>🍊 Fruit Taste Classifier 🍋</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Classify fruit taste (sweet or sour) from description</p>", unsafe_allow_html=True)
st.write("---")

# Input box
desc = st.text_area("Enter fruit description:", placeholder="e.g. It tastes tangy and juicy with a little sourness...")

# Predict button
if st.button("Predict Taste"):
    if desc.strip() == "":
        st.warning("Please enter a fruit description!")
    else:
        vec = vectorizer.transform([desc])
        prediction = model.predict(vec)[0]
        if prediction.lower() == "sweet":
            st.success("🍯 The fruit tastes **Sweet!**")
        elif prediction.lower() == "sour":
            st.error("🍋 The fruit tastes **Sour!**")
        else:
            st.info(f"The fruit tastes **{prediction}!**")

# Styling
st.markdown("""
<style>
    textarea {
        border-radius: 10px !important;
        font-size: 16px !important;
        padding: 10px !important;
    }
    .stButton>button {
        background-color: #FF6600;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #FF8533;
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# Footer
st.write("---")
st.markdown("<p style='text-align:center; color:gray;'>Developed by Shantanu Sharma 💻</p>", unsafe_allow_html=True)
