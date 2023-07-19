import streamlit as st
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
import plotly.express as px

# Load the model
model = load_model('text_generation_model.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Maximum length of sequence, adjust as necessary
max_length = 442  # Update this to the maximum length of sequence that you trained your model with

# Function to generate a sentence
def generate_sentence(model, tokenizer, max_length, seed_text, num_words, temperature):
    for _ in range(num_words):
        sequence = tokenizer.texts_to_sequences([seed_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length-1, padding='pre')
        probabilities = model.predict(sequence, verbose=0)
        
        # Apply temperature
        probabilities = np.asarray(probabilities).astype('float64')
        probabilities = np.log(probabilities) / temperature
        exp_preds = np.exp(probabilities)
        probabilities = exp_preds / np.sum(exp_preds)

        predicted = np.random.choice(range(len(probabilities[0])), p = probabilities.ravel())
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Set up the UI
st.title(" 🍔 McDeepNet: 20k McDonalds Review's 🍔 ")

# Form to take user inputs
with st.form(key='my_form'):
    seed_text = st.text_input(label='Enter the seed text for completion')
    num_words = st.number_input(label='Enter the number of words to generate', min_value=1, max_value=100, value=5)
    temperature = st.slider(label='Set temperature', min_value=0.1, max_value=3.0, value=1.0, step=0.1)
    submit_button = st.form_submit_button(label='Generate Text')

# Generate and display the output on form submission
if submit_button:
    sentence = generate_sentence(model, tokenizer, max_length, seed_text, num_words, temperature)
    st.write(sentence)
    
