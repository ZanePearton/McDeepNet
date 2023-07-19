# 🍔 McDeepNet: Trained on 20k McDonalds Review 🍔 

This project contains a Machine Learning (ML) model that is trained on 20,000 McDonald's reviews. The ML model uses Recurrent Neural Networks (RNNs) to learn patterns from the provided reviews and then, using this learning, generates a unique review of its own. The model can produce different types of output based on seed text and the 'temperature' set by the user.

The application is built using [Streamlit](https://streamlit.io), a fast and easy way to build data applications, which handles the user interface (UI) for taking user inputs and displaying the model's generated review. 

## How it works

This program reads in a seed text provided by the user along with the number of words to generate and the temperature. It then passes these inputs to the model to generate a review. 

The 'temperature' parameter helps in controlling the randomness of predictions by scaling the logits before applying softmax. With a higher value, the model's outputs will be more diverse. With a lower value, the model's outputs will be more focused on the most probable words.

Here's a simplified explanation of how the text generation function works:

1. The function receives the model, tokenizer, maximum sequence length, seed text, number of words to generate, and temperature as inputs.

2. The seed text is tokenized (converted into a numerical representation that the model can process) and padded to the maximum sequence length.

3. The model predicts the next word based on the seed text.

4. The output word is selected based on the model's probabilities and the temperature parameter. 

5. The output word is then appended to the seed text and the process repeats for the desired number of words.

## Requirements

The main requirements for running this application are:

- Python
- TensorFlow
- Streamlit
- Numpy
- Pickle

Please refer to `requirements.txt` for the complete list of dependencies.

## Running the application

You can run this application locally by using the following command:

```bash
streamlit run app.py
```

## Customization

You can adjust the `max_length` variable according to the maximum sequence length that you trained your model with. This is currently set to `442`.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
