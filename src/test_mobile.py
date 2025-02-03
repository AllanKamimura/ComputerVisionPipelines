import cv2
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter
from ai_edge_litert.interpreter import Delegate

# input
img = cv2.imread('frame.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_resized = tf.image.resize(img, [300,300], method='bicubic', preserve_aspect_ratio=False)
img_resized = tf.clip_by_value(img_resized, 0, 255)
img_resized = tf.cast(img_resized, dtype=tf.uint8)
img_resized = tf.transpose(img_resized, [2, 0, 1])
img_resized = tf.reshape(img_resized, [1, 300, 300, 3])
tensor = tf.convert_to_tensor(img_resized, dtype=tf.uint8)

# load model
model_path = 'lite-model_ssd_mobilenet_v1_1_metadata_2.tflite'
interpreter = Interpreter(
    model_path=model_path,
    experimental_delegates = [Delegate("/usr/lib/libvx_delegate.so")]
    )
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# inference
interpreter.set_tensor(input_details[0]['index'], tensor)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
output = output.reshape(224, 224)

# output file
prediction = cv2.resize(output, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
print(" Write image to: output.png")
depth_min = prediction.min()
depth_max = prediction.max()
img_out = (255 * (prediction - depth_min) / (depth_max - depth_min)).astype("uint8")

cv2.imwrite("output.png", img_out)
