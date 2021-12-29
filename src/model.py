import tensorflow as tf
from .layer import *
from .mask import *

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self._d_model = d_model
        self._num_layers = num_layers
        
        self._embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self._pos_encoding = positional_encoding(maximum_position_encoding, self._d_model)

        self._enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self._dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        x = self._embedding(x)
        x *= tf.math.sqrt(tf.cast(self._d_model, tf.float32))
        x += self._pos_encoding[:, :seq_len, :]
        x = self._dropout(x, training=training)

        for i in range(self._num_layers):
            x = self._enc_layers[i](x, training, mask)
        
        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self._d_model = d_model
        self._num_layers = num_layers
        
        self._embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self._pos_encoding = positional_encoding(maximum_position_encoding, self._d_model)

        self._dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self._dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self._embedding(x)
        x *= tf.math.sqrt(tf.cast(self._d_model, tf.float32))
        x += self._pos_encoding[:, :seq_len, :]
        x = self._dropout(x, training=training)

        for i in range(self._num_layers):
            x, block1, block2 = self._dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)

        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)

        return final_output, attention_weights