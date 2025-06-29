This project implements a quantum inspired attention mechanism using TensorFlow/Keras, extending the traditional Transformer architecture by introducing complex valued phase interactions and entanglement between attention heads.

We demonstrate its application on learning dynamics of the Lorenz attractor, a classic chaotic system.

Quantum phase attention:
1. Inspired by quantum interference, we model attention weights as complex valued functions instead of linear dot product in traditional Transformer architecture to learn complex relationships between tokens.
2. Real and imaginary parts arise from trigonometric (sin/cos) functions of the scaled dot product.
 
Entanglement Across Attention Heads:
1. Unlike standard multi-head attention with independant heads, this model allows cross head interaction via a learnable entanglement matrix.
2. This enables shared structure across heads, mimicking quantum entanglement behavior.

 The attention model was tested on synthetic data of the Lorenz attractor, a well-known chaotic dynamical system.
Despite the inherent unpredictability, the quantum-inspired attention learns short-term trajectory prediction effectively.\


Core components :
1. quantum_attention:
	•	d_model: total dimensionality
	•	num_heads: number of attention heads
	•	temperature: scaling factor for phase function
	•	use_entanglement: enable/disable head mixing
	•	phase_modulation: adaptive or fixed
	•	Returns:
	•	Transformed sequence output with skip connection and layer norm.


