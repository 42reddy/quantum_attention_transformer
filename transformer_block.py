import tensorflow as tf
import tensorflow.keras.layers
from tensorflow.keras import layers
from quantum_attention import quantum_attention


class transformer_block(tf.keras.layers.Layer):

    def __init__(self, d_model, d_k, ff_dim):
        """
        Initializes a quantum-enhanced transformer block with attention,
        normalization, feed-forward layers, and dropout.

        Args:
            d_model (int): Dimensionality of the token embedding.
            d_k (int): Dimensionality of the attention key/query vectors.
            ff_dim (int): Dimensionality of the feed-forward network's hidden layer.

        Attributes:
            attention: A custom quantum attention layer using d_model and d_k.
            norm1: Layer normalization applied after the attention mechanism.
            norm2: Layer normalization applied after the feed-forward network.
            ffn: A feed-forward network with one hidden layer (ReLU) and output layer.
            dropout1: Dropout applied after the attention output.
            dropout2: Dropout applied after the feed-forward output.
        """

        super(transformer_block, self).__init__()

        self.attention = quantum_attention(d_model, d_k)     # quantum attention layer

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)  # first layer of normalization
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.ffn = tensorflow.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(d_model)
        ])                                                  # Hidden feed forward network

        self.dropout1 = layers.Dropout(0.1)                 # first dropout layer
        self.dropout2 = layers.Dropout(0.1)


    def call(self, inputs, training=False):
        """
        Executes the forward pass of the transformer block.

        Applies quantum attention followed by residual connection and layer
        normalization, then a feed forward network with another residual and
        normalization step.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, d_model).
            training : Boolean if model is in training mode
                (controls dropout). Defaults is False.

        Returns:
            Tensor: Output tensor of the same shape as `inputs` after applying
            attention, feed-forward network, and normalization.
        """

        attn_out = self.attention(inputs)
        attn_out = self.dropout1(attn_out, training=training)
        out1 = self.norm1(inputs + attn_out)

        ffn_out = self.ffn(out1)
        ffn_output = self.dropout2(ffn_out, training=training)

        return self.norm2(out1 + ffn_output)





