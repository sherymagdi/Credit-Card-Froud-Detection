import streamlit as st
import re
import joblib
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already present
nltk.download('stopwords')

# # Load model and vectorizer
# vectorizer = joblib.load('./Models/Vectorizer.pkl')
# model = joblib.load('./Models/DecisionTreeClassification.pkl')

# Text preprocessing function
def preprocess_text(text_data):
    preprocessed_text = []
    for sentence in text_data:
        sentence = re.sub(r'[^\w\s]', '', sentence)
        preprocessed_text.append(' '.join(token.lower()
                                          for token in sentence.split()
                                          if token.lower() not in stopwords.words('english')))
    return preprocessed_text

# Streamlit App
st.set_page_config(page_title="Text Classifier", layout="centered")
st.title("🧠 Text Classification App")
st.write("Enter some text below and the model will classify it.")

user_input = st.text_area("Enter Text Here", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text before predicting.")
    else:
        processed = preprocess_text([user_input])
        # vect_text = vectorizer.transform(processed)
        # prediction = model.predict(vect_text)[0]
        st.success(f"🔍 Prediction: *{processed}*")