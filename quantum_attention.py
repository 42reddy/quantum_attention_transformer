import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class quantum_attention(tf.keras.layers.Layer):

    def __init__(self, d_model, d_k):

        super().__init__()
        self.d_k = d_k
        self.d_model = d_model
        self.W_q = layers.Dense(d_k)
        self.W_k = layers.Dense(d_k)
        self.W_v = layers.Dense(d_k)

    def attention(self, x):
        """
        Returns a tensor with attention functions for the tokens in the series based on the rest of the keys

        :param x: individual token in a feature vector
        :return: attention function for the input token based on all the keys of the feature vector
        """
        Q = self.W_q(x)     # Query matrix
        K = self.W_k(x)     # Key matrix
        V = self.W_v(x)     # Value matrix

        QK = tf.matmul(Q, K, transpose_b = True)

        complex_phase = tf.complex(tf.math.cos(QK), tf.math.sin(QK))    # a quantum wavefuntion equivalent containing all the information about the relationship of the tokens with each oother

        V_complex = tf.complex(V, tf.zeros_Like(V))   # the Value matrix expressed as a complex phase function

        output = tf.math.real(tf.matmul(complex_phase, V_complex))     # the analogus of the dot product in a classical attention block,
                                                                        # returns attention vector based on the rest of the keys introducing interference pattern among them
                                                                        # recovering complex relationships over the linear ones in classical attention block

        return output

    def position_encoding(self, seq_len, d_model):
        """
        Adds the sense of the position for the tokens in the timeseries

        :param seq_len: length of the feature vector
        :param d_model: length of the encoded vector for each token
        :return: position encoded tensor for the feature vector
        """

        position = np.arange(seq_len)[:, np.newaxis]
        div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(position * div)
        pe[:, 1::2] = np.cos(position * div)

        return tf.convert_to_tensor(pe, dtype=tf.float32)





