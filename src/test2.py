import cv2
import tensorflow as tf
import tflite_runtime.interpreter as tflite

# input
img = cv2.imread('frame.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

img_resized = tf.image.resize(img, [224,224], method='bicubic', preserve_aspect_ratio=False)
img_resized = tf.transpose(img_resized, [2, 0, 1])
img_resized = tf.reshape(img_resized, [1, 3, 224, 224])
tensor = tf.convert_to_tensor(img_resized, dtype=tf.float32)

# load model
model_path = '/home/torizon/MiDaS_small_pt2e.tflite'
interpreter = tflite.Interpreter(
    model_path=model_path,
    experimental_delegates = [tflite.load_delegate("/usr/lib/libvx_delegate.so")]
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
