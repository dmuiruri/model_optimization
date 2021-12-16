#! /usr/bin/env python

import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
import numpy as np
assert float(tf.__version__[:3]) >= 2.3


# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

def create_keras_model():
    """
    Create a keras model
    """
    # Define the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28)),
        tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
    ])
    
    # Train the digit classification model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])
    model.fit(
        train_images,
        train_labels,
        epochs=5,
        validation_data=(test_images, test_labels)
    )
    return model

def convert_to_tflite_model(keras_model):
    """Convert the Tensorflow model to a tensorflow lite model

    """
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    return tflite_model

def representative_data_gen():
    """A helper function to provide a representative dataset for
    quantization. The converter uses this to estimate the dynamic
    range of all variable data in the model.

    """
    for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
        # The model used here has only one input so each data point
        # has one element.
        yield [input_value]

def create_default_optimizations(model, variable_data=False):
    """Enable default optimizations, if any operation cannot be quantized,
    the default float32 is applied. In this quantization approach, the
    weights and variable data is quantized but the input and output
    layer is not.

    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if variable_data:
        converter.representative_dataset = representative_data_gen
    tflite_model_quant = converter.convert()
    return tflite_model_quant

def optimize_model_input_outputs(variable_data=False):
    """Convert the inputs and outputs of a model to uint8

    To quantize both weights and the variable data, set variable_data
    to True

    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if variable_data:
        converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model_quant = converter.convert()
    return tflite_model_quant

    
def input_ouput_datatype(tflite_model):
    """
    Check input and output data type

    Returns a tuple of (input, output)
    """
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    input_type = interpreter.get_input_details()[0]['dtype']
    output_type = interpreter.get_output_details()[0]['dtype']
    return (input_type, output_type)


tflite_model_quant = converter.convert()
if __name__ == '__main__':
    """Add unit tests here"""
    keras_model = create_keras_model()
    tflite_model = convert_to_tflite_model(keras_model)
    print(input_output_datatype(flite_model))
