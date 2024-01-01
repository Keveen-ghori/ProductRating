# streamlit_app.py
import streamlit as st
import pandas as pd
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split


# Load stopwords
english_stops = set(stopwords.words('english'))

def load_dataset():
    df = pd.read_csv('product..csv')
    x_data = df['review']       # Reviews/Input
    y_data = df['sentiment']    # Sentiment/Output

    # PRE-PROCESS REVIEW
    x_data = x_data.replace({'<.*?>': ''}, regex = True)          # remove html tag
    x_data = x_data.replace({'[^A-Za-z]': ' '}, regex = True)     # remove non alphabet
    x_data = x_data.apply(lambda review: [w for w in review.split() if w not in english_stops])  # remove stop words
    x_data = x_data.apply(lambda review: [w.lower() for w in review])   # lower case
    
    # ENCODE SENTIMENT -> 0 & 1
    y_data = y_data.replace('positive', 1)
    y_data = y_data.replace('negative', 0)

    return x_data, y_data

x_data, y_data = load_dataset()
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2)

def get_max_length():
    review_length = []
    for review in x_train:
        review_length.append(len(review))

    return int(np.ceil(np.mean(review_length)))

# ENCODE REVIEW
token = Tokenizer(lower=False)    # no need lower, because already lowered the data in load_data()
token.fit_on_texts(x_train)
x_train = token.texts_to_sequences(x_train)
x_test = token.texts_to_sequences(x_test)

max_length = get_max_length()

x_train = pad_sequences(x_train, maxlen=max_length, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=max_length, padding='post', truncating='post')

total_words = len(token.word_index) + 1

# ARCHITECTURE
EMBED_DIM = 32
LSTM_OUT = 64

model = Sequential()
model.add(Embedding(total_words, EMBED_DIM, input_length = max_length))
model.add(LSTM(LSTM_OUT))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

checkpoint = ModelCheckpoint(
    'models/LSTM.h5',
    monitor='accuracy',
    save_best_only=True,
    verbose=1
)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(x_train, y_train)

loaded_model = load_model('models/LSTM.h5')

# Initialize Tokenizer
token = Tokenizer(lower=False)  # no need lower, because already lowered the data in load_data()


# Function to preprocess input
def preprocess_input(review):
    regex = re.compile(r'[^a-zA-Z\s]')
    review = regex.sub('', review)
    words = review.split(' ')
    filtered = [w for w in words if w not in english_stops]
    filtered = ' '.join(filtered)
    filtered = [filtered.lower()]
    return filtered


# Function to predict sentiment
def predict_sentiment(review):
    preprocessed_review = preprocess_input(review)
    tokenize_words = token.texts_to_sequences([preprocessed_review])
    tokenize_words = pad_sequences(tokenize_words, maxlen=max_length, padding='post', truncating='post')
    result = loaded_model.predict(tokenize_words)
    return result



# Streamlit app
def main():
    st.title("Product Sentiment Analysis")

    # Input text area for user to enter a product review
    review_input = st.text_area("Enter your product review:")

    if st.button("Submit"):
        if review_input:
            # Predict sentiment and display rating
            sentiment_score = predict_sentiment(review_input)

            if sentiment_score >= 0.8:
                rating = '*****'
            elif sentiment_score > 0.65:
                rating = '****'
            elif sentiment_score > 0.5:
                rating = '***'
            elif sentiment_score > 0.3:
                rating = '**'
            else:
                rating = '*'

            st.subheader("Predicted Rating:")
            st.write(rating)

if __name__ == "__main__":
    main()
