from fastapi import FastAPI
import tensorflow as tf
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

# Add parent directory to sys.path to recognize 'training'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.model import Encoder, Decoder  # Import Encoder & Decoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Correct paths for tokenizer files
tokenizer_inputs_path = os.path.join(BASE_DIR, "tokenizer_inputs.pkl")
tokenizer_outputs_path = os.path.join(BASE_DIR, "tokenizer_outputs.pkl")

# Load Tokenizers
with open(tokenizer_inputs_path, 'rb') as f:
    tokenizer_inputs = pickle.load(f)

with open(tokenizer_outputs_path, 'rb') as f:
    tokenizer_outputs = pickle.load(f)
# Model Hyperparameters
HIDDEN_DIM = 256
EMBEDDING_DIM = 100
VOCAB_SIZE_INPUTS = len(tokenizer_inputs.word_index) + 1
VOCAB_SIZE_OUTPUTS = len(tokenizer_outputs.word_index) + 1

# Initialize Encoder & Decoder
encoder = Encoder(VOCAB_SIZE_INPUTS, EMBEDDING_DIM, HIDDEN_DIM)
decoder = Decoder(VOCAB_SIZE_OUTPUTS, EMBEDDING_DIM, HIDDEN_DIM)

# **IMPORTANT: Build the model before loading weights**
encoder(tf.zeros((1, 67)), encoder.init_states(1))
decoder(tf.zeros((1, 1)), encoder.init_states(1))

# Load Pretrained Weights
weights_path = os.path.join(BASE_DIR, ".weights.h5")
encoder.load_weights(weights_path)
decoder.load_weights(weights_path)

app = FastAPI()

@app.post("/translate/")
def translate_text(input_text: str):
    input_seq = tokenizer_inputs.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=67, padding='post')

    en_initial_states = encoder.init_states(1)
    en_outputs, en_state_h, en_state_c = encoder(tf.constant(input_seq), en_initial_states)

    de_input = tf.constant([[tokenizer_outputs.word_index['<sos>']]])
    de_state_h, de_state_c = en_state_h, en_state_c

    out_words = []
    for _ in range(20):
        de_output, de_state_h, de_state_c = decoder(de_input, (de_state_h, de_state_c))
        de_input = tf.argmax(de_output, -1)
        word = tokenizer_outputs.index_word[de_input.numpy()[0][0]]
        if word == "<eos>":
            break
        out_words.append(word)

    return {"translated_text": " ".join(out_words)}
