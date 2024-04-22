import re
import string
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

def custom_standardization(input_data):
    # Convert to lowercase
    lowercase = input_data.lower()
    # Remove HTML tags
    stripped_html = re.sub(r'<br\s*/?>', ' ', lowercase)
    # Remove URLs
    stripped_links = re.sub(r'https?://\S+|www\.\S+', '', stripped_html)
    # Remove square brackets and text within them
    stripped_brackets = re.sub(r'\[.*?\]', '', stripped_links)
    # Remove punctuation
    no_punctuation = stripped_brackets.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    no_numbers = re.sub(r'\w*\d\w*', '', no_punctuation)
    # Remove extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', no_numbers).strip()
    return cleaned_text

def preprocess_tweet(tweet, tokenizer, sequence_length):
    # Preprocess the tweet
    processed_tweet = custom_standardization(tweet)
    # Tokenize the tweet
    tokenized_tweet = tokenizer.texts_to_sequences([processed_tweet])
    # Pad the sequence
    padded_tweet = pad_sequences(tokenized_tweet, maxlen=sequence_length, padding='post', truncating='post')
    return np.array(padded_tweet)

def predict_sentiment(tweet, model, tokenizer, sequence_length):
    # Preprocess the tweet
    processed_tweet = preprocess_tweet(tweet, tokenizer, sequence_length)
    # Predict sentiment
    prediction = model.predict(processed_tweet)
    # Output result
    if prediction >= 0.5:
        return "Disaster"
    else:
        return "Not a Disaster"

def load_and_predict(tweet_input):
    # Load the saved model
    model = load_model("sentiment_model.h5")
    # Load the tokenizer
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    # Define sequence length
    sequence_length = 100
    # Predict sentiment
    result = predict_sentiment(tweet_input, model, tokenizer, sequence_length)
    return result

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        tweet_input = request.form['tweet']
        result = load_and_predict(tweet_input)
        return render_template('index.html', tweet=tweet_input, prediction=result)
    return render_template('index.html', tweet=None, prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
