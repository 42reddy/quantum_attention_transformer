import numpy as np
import tensorflow as tf
from quantum_transformer import Transformer
from data_generation import Data_Generator
from tqdm import tqdm


T = 1000     # Total simulation time
dt = 0.001   # timestep
input_len = 50  # input feature vector to the transformer
pred_length = 10  # output sequence length
batch_size = 32

generator = Data_Generator()
data = generator.Lorenz_attractor(T, dt)           # Generates chaotic Lorenz time series
X, y = generator.sequences(data, input_len, pred_length)    # Generates training and output sequences
X, y = generator.normalize(X, y)                   # normalize

""" Generate a tensorflow dataset and split into training and validation sets """
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_val, y_val = X[split:], y[split:]

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(batch_size).prefetch(1)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(1)


d_model = 128      # Transformer embedding dimension
num_outputs = 3    # output dimension, 3 in chaotic Lorenz attractor

epochs = 10        # number of training epochs
batch_size = 64
learning_rate = 1e-4

model = Transformer(
    n_layers=5,
    d_model=128,
    d_k=64,
    ff_dim=256,
    seq_len=X_train.shape[1],
    output_dim=y_train.shape[2],
)     # initilizes the custom transformer model
dummy_input = tf.zeros((1, 50, 3))  # (batch, seq_len, input_dim)
model(dummy_input)  # Initiate arbitrary weights
model.summary()


loss_function = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.legacy.Adam()

for epoch in range(epochs):

    print(f"epoch{epoch}/{epochs}")

    epoch_loss = 0
    batches = 0

    for step, (x_batch, y_batch) in enumerate(tqdm(train_dataset, desc="Training")):
        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training =True)
            loss = loss_function(y_batch, y_pred)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoch_loss += loss.numpy()
        batches += 1

    print(f"Loss: {epoch_loss / batches:.6f}")



def evaluate_model(model, test_dataset, loss_function):
    total_loss = 0.0
    total_batches = 0

    for step, (x_batch, y_batch) in enumerate(test_dataset):
        y_pred = model(x_batch, training=False)
        loss = loss_function(y_batch, y_pred)
        total_loss += loss.numpy()
        total_batches += 1

    avg_loss = total_loss / total_batches
    print(f"\nEvaluation Loss: {avg_loss:.6f}")
    return avg_loss

test_loss = evaluate_model(model, val_dataset, loss_function)

mae_metric = tf.keras.metrics.MeanAbsoluteError()

for x_batch, y_batch in val_dataset:
    y_pred = model(x_batch, training=False)
    mae_metric.update_state(y_batch, y_pred)

print("MAE:", mae_metric.result().numpy())

