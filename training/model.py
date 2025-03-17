import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.lstm = LSTM(hidden_dim, return_state=True, return_sequences=True)

    def call(self, inputs, states):
        x = self.embedding(inputs)
        outputs, state_h, state_c = self.lstm(x, initial_state=states)
        return outputs, state_h, state_c

    def init_states(self, batch_size):
        return (tf.zeros((batch_size, self.lstm.units)), tf.zeros((batch_size, self.lstm.units)))

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.lstm = LSTM(hidden_dim, return_state=True, return_sequences=True)
        self.dense = Dense(vocab_size, activation='softmax')

    def call(self, inputs, states):
        x = self.embedding(inputs)
        outputs, state_h, state_c = self.lstm(x, initial_state=states)
        outputs = self.dense(outputs)
        return outputs, state_h, state_c
