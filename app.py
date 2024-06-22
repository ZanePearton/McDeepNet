
# library imports
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from collections import Counter

# Load the model
model = load_model('text_generation_model.h5', custom_objects={'Orthogonal': tf.keras.initializers.Orthogonal()})

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

# Function to create a tree diagram
def create_tree_diagram(data):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes and edges to the graph
    for i, (word, prob) in enumerate(data):
        G.add_node(i, label=word, probability=prob)
        if i > 0:
            G.add_edge(i - 1, i)

    # Get node positions for the layout
    pos = nx.spring_layout(G)

    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{G.nodes[node]['label']} ({G.nodes[node]['probability']:.2f})")
        node_color.append(G.nodes[node]['probability'])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            color=node_color,
            colorbar=dict(
                thickness=15,
                title='Probability',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Text Generation Tree Diagram',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False)
                    ))
    return fig

# Set up the UI
st.title("🍔 McDeepNet 🍔")
st.subheader("Trained on 20k McDonald's Reviews")
st.caption("Welcome to McDeepNet! This project uses a Machine Learning (ML) model trained on 20,000 McDonald's reviews. It's an interesting application that employs Recurrent Neural Networks (RNNs) to learn patterns from these reviews and, subsequently, generates a unique review of its own. The model can produce varying types of output based on a seed text and a temperature parameter provided by the user. Checkout my github: https://github.com/zanepearton ")

# Form to take user inputs
with st.form(key='my_form'):
    seed_text = st.text_input(label='Enter the seed text for sentence completion')
    num_words = st.number_input(label='Enter the number of words to generate', min_value=1, max_value=100, value=50)
    temperature = st.slider(label='Set temperature', min_value=0.1, max_value=3.0, value=1.0, step=0.1)
    submit_button = st.form_submit_button(label='Generate Text')

# Generate and display the output on form submission
if submit_button:
    sentence, word_probs = generate_sentence(model, tokenizer, max_length, seed_text, num_words, temperature)
    # st.write(word_probs)
    st.write(sentence)
    
    # Count word frequencies
    word_freq = Counter(sentence.split())
    # Create and display the tree diagram
    fig_tree = create_tree_diagram(word_probs)
    st.plotly_chart(fig_tree)

    
    # Create a DataFrame for the frequencies
    freq_df = pd.DataFrame(list(word_freq.items()), columns=['Word', 'Frequency'])
    
    # Create a Plotly Express bar chart
    fig_bar = px.bar(freq_df, x='Word', y='Frequency', title='Word Frequencies')
    st.plotly_chart(fig_bar)

# library imports
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from collections import Counter

# Load the model
model = load_model('text_generation_model.h5', custom_objects={'Orthogonal': tf.keras.initializers.Orthogonal()})

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Maximum length of sequence, adjust as necessary
max_length = 442  # Update this to the maximum length of sequence that you trained your model with

# Function to generate a sentence with context
def generate_sentence(model, tokenizer, max_length, seed_text, num_words, temperature):
    word_probs = []
    context = seed_text.split()[-3:]  # Get the last three words as initial context
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
        context = (context + [output_word])[-3:]  # Update context with the new word
        word_probs.append((output_word, probabilities[0, predicted], context.copy()))
    return seed_text, word_probs

# Function to create a tree diagram with clearer context
def create_tree_diagram(data):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes and edges to the graph
    for i, (word, prob, context) in enumerate(data):
        main_node = f"main_{i}"
        G.add_node(main_node, label=word, probability=prob)
        
        # Add context nodes and edges
        for j, ctx_word in enumerate(context):
            ctx_node = f"ctx_{i}_{j}"
            G.add_node(ctx_node, label=ctx_word, probability=0)  # Context nodes have no probability
            G.add_edge(ctx_node, main_node)  # Connect context word to current main word

    # Get node positions for the layout
    pos = nx.spring_layout(G)

    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{G.nodes[node]['label']} ({G.nodes[node]['probability']:.2f})" if G.nodes[node]['probability'] > 0 else G.nodes[node]['label'])
        node_color.append(G.nodes[node]['probability'])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            color=node_color,
            colorbar=dict(
                thickness=15,
                title='Probability',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Text Generation Tree Diagram',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False)
                    ))
    return fig

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
    st.write(sentence)
    
    # Count word frequencies
    word_freq = Counter(sentence.split())
    # Create and display the tree diagram
    fig_tree = create_tree_diagram(word_probs)
    st.plotly_chart(fig_tree)
    
    # Create a DataFrame for the frequencies
    freq_df = pd.DataFrame(list(word_freq.items()), columns=['Word', 'Frequency'])
    
    # Create a Plotly Express bar chart
    fig_bar = px.bar(freq_df, x='Word', y='Frequency', title='Word Frequencies')
    st.plotly_chart(fig_bar)
# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import plotly.express as px
# import plotly.graph_objects as go
# import networkx as nx
# from collections import Counter

# # Load the model
# model = load_model('text_generation_model.h5', custom_objects={'Orthogonal': tf.keras.initializers.Orthogonal()})

# # Load the tokenizer
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)

# # Maximum length of sequence, adjust as necessary
# max_length = 442  # Update this to the maximum length of sequence that you trained your model with

# # Function to generate a sentence with context
# def generate_sentence(model, tokenizer, max_length, seed_text, num_words, temperature):
#     word_probs = []
#     context = seed_text.split()[-3:]  # Get the last three words as initial context
#     for _ in range(num_words):
#         sequence = tokenizer.texts_to_sequences([seed_text])[0]
#         sequence = pad_sequences([sequence], maxlen=max_length-1, padding='pre')
#         probabilities = model.predict(sequence, verbose=0)[0]

#         # Apply temperature
#         probabilities = np.asarray(probabilities).astype('float64')
#         probabilities = np.log(probabilities) / temperature
#         exp_preds = np.exp(probabilities)
#         probabilities = exp_preds / np.sum(exp_preds)

#         predicted = np.random.choice(range(len(probabilities)), p=probabilities)
#         output_word = ""
#         for word, index in tokenizer.word_index.items():
#             if index == predicted:
#                 output_word = word
#                 break
#         seed_text += " " + output_word
#         context = (context + [output_word])[-3:]  # Update context with the new word
#         word_probs.append((output_word, probabilities[predicted], context.copy()))
#     return seed_text, word_probs

# def create_tree_diagram(data):
#     # Create a directed graph
#     G = nx.DiGraph()

#     # Add nodes and edges to the graph
#     for i, (word, prob, context) in enumerate(data):
#         main_node = f"main_{i}"
#         G.add_node(main_node, label=word, probability=prob, is_main=True)
        
#         # Add context nodes and edges
#         for j, ctx_word in enumerate(context):
#             ctx_node = f"ctx_{i}_{j}"
#             G.add_node(ctx_node, label=ctx_word, probability=0, is_main=False)  # Context nodes have no probability
#             G.add_edge(ctx_node, main_node)  # Connect context word to current main word

#     # Get node positions for the layout
#     pos = nx.spring_layout(G)

#     # Create edge traces
#     edge_x = []
#     edge_y = []
#     for edge in G.edges():
#         x0, y0 = pos[edge[0]]
#         x1, y1 = pos[edge[1]]
#         edge_x.append(x0)
#         edge_x.append(x1)
#         edge_x.append(None)
#         edge_y.append(y0)
#         edge_y.append(y1)
#         edge_y.append(None)

#     edge_trace = go.Scatter(
#         x=edge_x, y=edge_y,
#         line=dict(width=2, color='#888'),
#         hoverinfo='none',
#         mode='lines')

#     # Create node traces
#     node_x = []
#     node_y = []
#     node_text = []
#     node_color = []
#     node_size = []
#     for node in G.nodes():
#         x, y = pos[node]
#         node_x.append(x)
#         node_y.append(y)
#         node_text.append(f"{G.nodes[node]['label']} ({G.nodes[node]['probability']:.2f})" if G.nodes[node]['probability'] > 0 else G.nodes[node]['label'])
#         if G.nodes[node]['is_main']:
#             node_color.append(G.nodes[node]['probability'])
#             node_size.append(15)
#         else:
#             node_color.append(0)  # Darker nodes for context words
#             node_size.append(10)

#     node_trace = go.Scatter(
#         x=node_x, y=node_y,
#         mode='markers+text',
#         text=node_text,
#         marker=dict(
#             showscale=True,
#             colorscale='YlGnBu',
#             size=node_size,
#             color=node_color,
#             colorbar=dict(
#                 thickness=15,
#                 title='Probability',
#                 xanchor='left',
#                 titleside='right'
#             ),
#             line_width=2))

#     # Create the figure
#     fig = go.Figure(data=[edge_trace, node_trace],
#                     layout=go.Layout(
#                         title='Text Generation Tree Diagram',
#                         titlefont_size=16,
#                         showlegend=False,
#                         hovermode='closest',
#                         margin=dict(b=20, l=5, r=5, t=40),
#                         xaxis=dict(showgrid=False, zeroline=False),
#                         yaxis=dict(showgrid=False, zeroline=False)
#                     ))
#     return fig


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
#     st.write(sentence)
    
#     # Count word frequencies
#     word_freq = Counter(sentence.split())
#     # Create and display the tree diagram
#     fig_tree = create_tree_diagram(word_probs)
#     st.plotly_chart(fig_tree)
    
#     # Create a DataFrame for the frequencies
#     freq_df = pd.DataFrame(list(word_freq.items()), columns=['Word', 'Frequency'])
    
#     # Create a Plotly Express bar chart
#     fig_bar = px.bar(freq_df, x='Word', y='Frequency', title='Word Frequencies')
#     st.plotly_chart(fig_bar)
