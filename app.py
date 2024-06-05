import re
import nltk
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import hashlib

# Load necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the model and tokenizer
model = load_model('text_classification_model.keras')
tokenizer = joblib.load('tokenizer.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# Load the max_length value
max_length_path = 'max_length.pkl'
with open(max_length_path, 'rb') as handle:
    max_length = pickle.load(handle)

# Load the stored MD5 checksum of the trained model
with open('model_md5.txt', 'r') as f:
    stored_md5 = f.read()

# Calculate MD5 checksum of the loaded model
loaded_model_md5 = hashlib.md5(open('text_classification_model.keras', 'rb').read()).hexdigest()

# Compare the calculated MD5 checksum with the stored MD5 checksum
if loaded_model_md5 != stored_md5:
    st.error("Error: MD5 checksum of the loaded model does not match the stored MD5 checksum.")
    st.stop()

# Stopwords
stop_words = set(stopwords.words('english'))

# Preprocess function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    word_tokens = word_tokenize(text)
    filtered_words = [word for word in word_tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    return ' '.join(lemmatized_words)

st.title('Title Categorization App')
st.write("Upload a CSV file with a column named 'titles' or enter a title for categorization.")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'titles' in df.columns:
        df['processed_title'] = df['titles'].apply(preprocess_text)
        sequences = tokenizer.texts_to_sequences(df['processed_title'])
        padded_sequences = pad_sequences(sequences, padding='post', maxlen=max_length)
        predictions = model.predict(padded_sequences)
        predicted_labels = label_encoder.inverse_transform(predictions.argmax(axis=1))
        df['predicted_category'] = predicted_labels
        st.write(df[['titles', 'predicted_category']])
        st.download_button(label="Download Predictions", data=df.to_csv(index=False), mime='text/csv', file_name='predictions.csv')
    else:
        st.error("The uploaded CSV file must contain a column named 'titles'.")

# Single title input
title_input = st.text_input("Or enter a title here")
if st.button("Categorize"):
    if title_input:
        processed_title = preprocess_text(title_input)
        sequence = tokenizer.texts_to_sequences([processed_title])
        padded_sequence = pad_sequences(sequence, padding='post', maxlen=max_length)
        prediction = model.predict(padded_sequence)
        predicted_label = label_encoder.inverse_transform(prediction.argmax(axis=1))
        st.write(f"The predicted category is: {predicted_label[0]}")
    else:
        st.error("Please enter a title.")

