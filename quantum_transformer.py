import tensorflow as tf
from tensorflow.keras import  layers
from quantum_attention import quantum_attention
from transformer_block import transformer_block



class Transformer(tf.keras.Model):

    def __init__(self, n_layers, d_model, d_k, ff_dim, seq_len, output_dim):
        """
        Initializes a Transformer model with custon quantum attention.

        Args:
            n_layers (int): Number of transformer blocks.
            d_model (int): Model embedding dimension.
            d_k (int): Attention key/query dimension.
            ff_dim (int): Size of the hidden feed-forward network.
            seq_len (int): Length of the input sequence.
            output_dim (int): Dimension of the final output.
        """
        super(Transformer, self).__init__()

        self.seq_len = seq_len
        self.d_model = d_model

        # Input embedding
        self.embedding = layers.Dense(d_model)

        # Positional encoding from quantum attention
        self.pos_embedding = self.get_positional_encoding(seq_len, d_model)  # shape: (1, seq_len, d_model)

        # Transformer blocks (with quantum attention inside)
        self.transformer_layers = [
            transformer_block(d_model, d_k, ff_dim) for _ in range(n_layers)
        ]

        # Output projection
        self.out = layers.Dense(output_dim)

    def call(self, x, training=False):

        x = self.embedding(x) + self.pos_embedding
        for block in self.transformer_layers:
            x = block(x, training=training)
        x = self.out(x)

        return x[:, :10, :]


    def get_positional_encoding(self, seq_len, d_model):
        """
        Generates sinusoidal positional encoding for input sequences. Returns a vector with the same dimension of d_model
         which Introduces a sense of position which is not inherent to the transformer architecture which processes all the
         tokens in parallel at once

        Args:
            seq_len (int): Length of the input sequence.
            d_model (int): Embedding dimension.

        Returns:
            Tensor: Positional encoding tensor of shape (1, seq_len, d_model).
        """

        position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]

        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = position * angle_rates

        # Apply sin to even indices, cos to odd
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)

        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)





