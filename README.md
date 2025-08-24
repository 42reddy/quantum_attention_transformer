# Quantum Attention Transformer

A quantum-inspired attention mechanism that incorporates complex phase functions and locality bias for enhanced sequence modeling.

## Overview

This implementation extends traditional self-attention by introducing quantum mechanical concepts:
- Complex-valued phase functions using trigonometric operations
- Multi-frequency oscillations (cos/sin and cos2/sin2 components)
- Learnable phase scaling parameters per attention head
- Quantum-inspired wavefunction evolution patterns

## Key Features

- **Quantum Phase Function**: Uses complex exponentials with cos/sin components to model quantum-like interactions
- **Multi-Head Architecture**: Each head has independent phase scaling parameters
- **Locality Bias**: Configurable locality scaling for position-aware attention
- **Temperature Control**: Adjustable temperature parameter for attention sharpness
- **Residual Connections**: Layer normalization with skip connections

## Usage

```python
import tensorflow as tf
from quantum_attention import quantum_attention

# Initialize the layer
attention_layer = quantum_attention(
    d_model=512,
    num_heads=8,
    temperature=1.0,
    locality_scale=2.0,
    dropout_rate=0.1
)

# Use in a model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, d_model),
    attention_layer,
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
```

## Parameters

- `d_model`: Model dimension (must be divisible by num_heads)
- `num_heads`: Number of attention heads
- `temperature`: Temperature scaling for attention weights
- `locality_scale`: Scaling factor for locality bias
- `dropout_rate`: Dropout probability during training

## Architecture

The quantum attention mechanism processes sequences through:

1. **Linear Projections**: Standard Q, K, V transformations
2. **Quantum Phase Computation**: Complex phase functions with learnable scaling
3. **Multi-Head Processing**: Independent quantum phases per head
4. **Attention Weighting**: Real components used for attention weights
5. **Complex Value Processing**: Both real and imaginary components contribute to output

## Requirements

```
tensorflow >= 2.0
numpy
```

## Implementation Notes

- Complex operations are handled using TensorFlow's native complex number support
- The imaginary component contribution is scaled by 0.1 for stability
- Phase scaling parameters are initialized to 1.0 and learned during training
- Compatible with standard transformer architectures as a drop-in replacement


