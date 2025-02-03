import tensorflow as tf
import numpy as np

from time import time

# Path to the SavedModel directory
saved_model_path = "ssd-mobilenet-v2-tensorflow2-ssd-mobilenet-v2"

# Load the SavedModel
model = tf.saved_model.load(saved_model_path)

# Get the signature for inference (adjust signature key if necessary)
infer = model.signatures["serving_default"]

for i in range(400):
    # Prepare input data as uint8 (adjust the shape to match your model's input shape)
    input_shape = (1, 300, 300, 3) 
    input_data = np.random.random_sample(input_shape).astype(np.uint8)

    # Create input tensor dictionary (adjust input tensor name if necessary)
    input_name = list(infer.structured_input_signature[1].keys())[0]
    input_tensor = tf.convert_to_tensor(input_data)
    t1 = time()
    # Perform inference
    output = infer(input_tensor)
    t2 = time()
    # Get the output tensor name and data
    output_name = list(output.keys())[0]
    output_data = output[output_name].numpy()

    # Print the results
    print(t2-t1)