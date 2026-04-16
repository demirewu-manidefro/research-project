import os
import tensorflow as tf
import pickle

MODEL_PATH = "../amharic_sentiment_lstm_model.keras"
TOKENIZER_PATH = "../tokenizer.pickle"

print(f"Checking {MODEL_PATH}: {os.path.exists(MODEL_PATH)}")
print(f"Checking {TOKENIZER_PATH}: {os.path.exists(TOKENIZER_PATH)}")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded.")
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("Tokenizer loaded.")
except Exception as e:
    print(f"Error: {e}")
