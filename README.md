# Model Optimization

This repo contains examples of model optimization strategies such as
quantization and prunning. Model optimization is increasingly
important in edge and IoT settins where devices have memory
constraints and low computing power.

## Quantization

Quantization reduces computations and model size  by reducing the precision of the
datatype from float32 bits precision to lower precions such as float16 or int8. Weights, biases, and activations may be quantized typically
to 8-bit integers although lower bit width implementations are also
discussed including binary neural networks.

## Prunning

# Reference
A survery paper on [Pruning and Quantization for Deep Neural Network Acceleration: A Survey](https://arxiv.org/abs/2101.09671)
