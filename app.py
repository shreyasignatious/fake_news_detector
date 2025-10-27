# app.py
import streamlit as st
import pickle

# Load saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detector")
st.markdown("Detect whether a news headline or article is **Real** or **Fake** using AI 🤖")

user_input = st.text_area("Enter a news headline or short article:")

if st.button("Check Authenticity"):
    if user_input.strip() == "":
        st.warning("Please enter a news headline or article.")
    else:
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        if prediction == 1:
            st.success("✅ This looks like **Real News!**")
        else:
            st.error("❌ This might be **Fake News!**")

st.caption("Developed by [Your Name] as part of Internship Project")
