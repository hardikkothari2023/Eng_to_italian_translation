import streamlit as st
import requests

st.title("English-to-Italian Translator")

input_text = st.text_area("Enter text in English:")

if st.button("Translate"):
    response = requests.post("http://localhost:8000/translate/", json={"input_text": input_text})
    if response.status_code == 200:
        st.write("### Translated Text:")
        st.write(response.json()["translated_text"])
    else:
        st.write("Error: Unable to get translation.")

st.write("Made with ❤️ by Hardik")