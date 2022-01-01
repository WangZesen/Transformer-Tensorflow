import numpy as np
import tensorflow as tf

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model
    )
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.sin(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq_len):
    mask = 1 - tf.sequence_mask(seq_len, dtype=tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(max_seq_len):
    mask = 1 - tf.linalg.band_part(tf.ones([max_seq_len, max_seq_len]), -1, 0)
    return mask

def create_masks(inp_len, tar_len, tar):
    enc_padding_mask = create_padding_mask(inp_len)
    dec_padding_mask = create_padding_mask(inp_len)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar_len)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask
