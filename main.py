import os
import re

import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
# py -3.9 -m streamlit run main.py use this version to run the app

MAX_FEATURES = 10000  # vocabulary size used when the model was trained
MAX_LEN = 500         # review length used when the model was trained
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simple_rnn_imdb.h5")


# The model was saved with a newer Keras that adds `quantization_config` to
# some layers; older Keras versions reject it, so drop it when loading.
class CompatEmbedding(Embedding):
    @classmethod
    def from_config(cls, config):
        config.pop("quantization_config", None)
        return super().from_config(config)


class CompatDense(Dense):
    @classmethod
    def from_config(cls, config):
        config.pop("quantization_config", None)
        return super().from_config(config)


@st.cache_resource
def load_resources():
    word_index = imdb.get_word_index()
    # compile=False: we only run inference, so no optimizer/loss is needed
    model = load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={"Embedding": CompatEmbedding, "Dense": CompatDense},
    )
    return word_index, model


def clean_text(text):
    # Lower-case, remove HTML tags and punctuation, return a list of words.
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)


def preprocess_text(text, word_index):
    encoded_review = []
    for word in clean_text(text):
        idx = word_index.get(word)
        if idx is None or idx + 3 >= MAX_FEATURES:
            encoded_review.append(2)  # unknown / out-of-vocabulary token
        else:
            encoded_review.append(idx + 3)  # same +3 offset used in training
    return sequence.pad_sequences([encoded_review], maxlen=MAX_LEN)


def predict_sentiment(review, word_index, model):
    preprocessed_input = preprocess_text(review, word_index)
    prediction = model.predict(preprocessed_input, verbose=0)
    score = float(prediction[0][0])
    sentiment = "Positive" if score > 0.5 else "Negative"
    return sentiment, score


st.set_page_config(page_title="IMDB Sentiment Classifier", page_icon="🎬")
st.title("🎬 IMDB Movie Review Sentiment Classification")
st.write("Enter a movie review to classify it as **Positive** or **Negative** using a Simple RNN.")

word_index, model = load_resources()

user_input = st.text_area("Movie Review", height=180)

if st.button("Classify"):
    if not user_input.strip():
        st.warning("Please enter a movie review.")
    else:
        sentiment, score = predict_sentiment(user_input, word_index, model)
        if sentiment == "Positive":
            st.success(f"Sentiment: {sentiment}")
        else:
            st.error(f"Sentiment: {sentiment}")
        st.write(f"Prediction score: {score:.4f}")
        st.progress(score)
else:
    st.info("Type a review above and click **Classify**.")
