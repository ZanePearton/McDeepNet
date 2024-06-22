# # library imports
# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import plotly.express as px
# from collections import Counter

# # Load the model
# model = load_model('text_generation_model.h5')

# # Load the tokenizer
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)

# # Maximum length of sequence, adjust as necessary
# max_length = 442  # Update this to the maximum length of sequence that you trained your model with

# # Function to generate a sentence
# def generate_sentence(model, tokenizer, max_length, seed_text, num_words, temperature):
#     word_probs = []
#     for _ in range(num_words):
#         sequence = tokenizer.texts_to_sequences([seed_text])[0]
#         sequence = pad_sequences([sequence], maxlen=max_length-1, padding='pre')
#         probabilities = model.predict(sequence, verbose=0)
        
#         # Apply temperature
#         probabilities = np.asarray(probabilities).astype('float64')
#         probabilities = np.log(probabilities) / temperature
#         exp_preds = np.exp(probabilities)
#         probabilities = exp_preds / np.sum(exp_preds)

#         predicted = np.random.choice(range(len(probabilities[0])), p = probabilities.ravel())
#         output_word = ""
#         for word, index in tokenizer.word_index.items():
#             if index == predicted:
#                 output_word = word
#                 break
#         seed_text += " " + output_word
#         word_probs.append((output_word, probabilities[0, predicted]))
#     return seed_text, word_probs

# # Set up the UI
# st.title("🍔 McDeepNet 🍔")
# st.subheader("Trained on 20k McDonald's Reviews")
# st.caption("Welcome to McDeepNet! This project uses a Machine Learning (ML) model trained on 20,000 McDonald's reviews. It's an interesting application that employs Recurrent Neural Networks (RNNs) to learn patterns from these reviews and, subsequently, generates a unique review of its own. The model can produce varying types of output based on a seed text and a temperature parameter provided by the user. Checkout my github: https://github.com/zanepearton ")

# # Form to take user inputs
# with st.form(key='my_form'):
#     seed_text = st.text_input(label='Enter the seed text for sentence completion')
#     num_words = st.number_input(label='Enter the number of words to generate', min_value=1, max_value=100, value=5)
#     temperature = st.slider(label='Set temperature', min_value=0.1, max_value=3.0, value=1.0, step=0.1)
#     submit_button = st.form_submit_button(label='Generate Text')

# # Generate and display the output on form submission
# if submit_button:
#     sentence, word_probs = generate_sentence(model, tokenizer, max_length, seed_text, num_words, temperature)
#     st.write(word_probs)
#     st.write(sentence)
    
#     # # Create dataframe to hold word probabilities and display in Streamlit
#     # prob_df = pd.DataFrame(word_probs, columns=["Word", "Probability"])
#     # st.dataframe(prob_df)

#     # Count word frequencies
#     word_freq = Counter(sentence.split())
    
#     # Create a DataFrame for the frequencies
#     freq_df = pd.DataFrame(list(word_freq.items()), columns=['Word', 'Frequency'])
    
#     # Create a Plotly Express bar chart
#     fig = px.bar(freq_df, x='Word', y='Frequency', title='Word Frequencies')
#     # fig = px.scatter_3d(freq_df, x='1', y='1', z='1')
#     # fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
#     #           color='green')
#     # Display the chart



# Library imports
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import Orthogonal

import plotly.express as px
from collections import Counter

# Function to load model with custom objects
def load_model_with_custom_objects(model_path):
    from tensorflow.keras.initializers import Orthogonal
    custom_objects = {'Orthogonal': Orthogonal}
    return load_model(model_path, custom_objects=custom_objects)

# Load the model
model = load_model_with_custom_objects('text_generation_model.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Maximum length of sequence, adjust as necessary
max_length = 442  # Update this to the maximum length of sequence that you trained your model with

# Function to generate a sentence
def generate_sentence(model, tokenizer, max_length, seed_text, num_words, temperature):
    word_probs = []
    for _ in range(num_words):
        sequence = tokenizer.texts_to_sequences([seed_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length-1, padding='pre')
        probabilities = model.predict(sequence, verbose=0)
        
        # Apply temperature
        probabilities = np.asarray(probabilities).astype('float64')
        probabilities = np.log(probabilities) / temperature
        exp_preds = np.exp(probabilities)
        probabilities = exp_preds / np.sum(exp_preds)

        predicted = np.random.choice(range(len(probabilities[0])), p=probabilities.ravel())
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
        word_probs.append((output_word, probabilities[0, predicted]))
    return seed_text, word_probs

# Set up the UI
st.title("🍔 McDeepNet 🍔")
st.subheader("Trained on 20k McDonald's Reviews")
st.caption("Welcome to McDeepNet! This project uses a Machine Learning (ML) model trained on 20,000 McDonald's reviews. It's an interesting application that employs Recurrent Neural Networks (RNNs) to learn patterns from these reviews and, subsequently, generates a unique review of its own. The model can produce varying types of output based on a seed text and a temperature parameter provided by the user. Checkout my github: https://github.com/zanepearton ")

# Form to take user inputs
with st.form(key='my_form'):
    seed_text = st.text_input(label='Enter the seed text for sentence completion')
    num_words = st.number_input(label='Enter the number of words to generate', min_value=1, max_value=100, value=5)
    temperature = st.slider(label='Set temperature', min_value=0.1, max_value=3.0, value=1.0, step=0.1)
    submit_button = st.form_submit_button(label='Generate Text')

# Generate and display the output on form submission
if submit_button:
    sentence, word_probs = generate_sentence(model, tokenizer, max_length, seed_text, num_words, temperature)
    st.write(word_probs)
    st.write(sentence)
    
    # Count word frequencies
    word_freq = Counter(sentence.split())
    
    # Create a DataFrame for the frequencies
    freq_df = pd.DataFrame(list(word_freq.items()), columns=['Word', 'Frequency'])
    
    # Create a Plotly Express bar chart
    fig = px.bar(freq_df, x='Word', y='Frequency', title='Word Frequencies')
    st.plotly_chart(fig)
