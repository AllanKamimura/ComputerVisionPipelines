import sys
import numpy as np
from time import time
import os
import cv2
import tflite_runtime.interpreter as tf  # TensorFlow Lite

# Read Environment Variables
USE_HW_ACCELERATED_INFERENCE = os.environ.get("USE_HW_ACCELERATED_INFERENCE", "1") == "1"
MINIMUM_SCORE = float(os.environ.get("MINIMUM_SCORE", 0.55))
CAPTURE_DEVICE = os.environ.get("CAPTURE_DEVICE", "/dev/video0")
CAPTURE_RESOLUTION_X = int(os.environ.get("CAPTURE_RESOLUTION_X", 640))
CAPTURE_RESOLUTION_Y = int(os.environ.get("CAPTURE_RESOLUTION_Y", 480))
CAPTURE_FRAMERATE = int(os.environ.get("CAPTURE_FRAMERATE", 30))

# Helper function to draw bounding boxes
def draw_bounding_boxes(img, labels, x1, x2, y1, y2, object_class):
    box_colors = [
        (254, 153, 143), (253, 156, 104), (253, 157, 13), (252, 204, 26),
        (254, 254, 51), (178, 215, 50), (118, 200, 60), (30, 71, 87),
        (1, 48, 178), (59, 31, 183), (109, 1, 142), (129, 14, 64)
    ]
    text_colors = [(0, 0, 0)] * 8 + [(255, 255, 255)] * 4

    cv2.rectangle(img, (x2, y2), (x1, y1), box_colors[object_class % len(box_colors)], 2)
    cv2.rectangle(img, (x1 + len(labels[object_class]) * 10, y1 + 15), (x1, y1), box_colors[object_class % len(box_colors)], -1)
    cv2.putText(img, labels[object_class], (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_colors[object_class % len(text_colors)], 1, cv2.LINE_AA)

def main():
    # Set up video capture
    cap = cv2.VideoCapture(CAPTURE_DEVICE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_RESOLUTION_X)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_RESOLUTION_Y)
    cap.set(cv2.CAP_PROP_FPS, CAPTURE_FRAMERATE)

    # Load labels
    with open("labelmap.txt", "r") as file:
        labels = file.read().splitlines()

    # Set up the interpreter
    delegates = [tf.load_delegate("/usr/lib/libvx_delegate.so")] if USE_HW_ACCELERATED_INFERENCE else []
    interpreter = tf.Interpreter(model_path="lite-model_ssd_mobilenet_v1_1_metadata_2.tflite", experimental_delegates=delegates)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]['shape'][1]

    while True:
        ret, image_original = cap.read()
        if not ret:
            break

        height1, width1 = image_original.shape[:2]
        image = cv2.resize(image_original, (input_size, int(input_size * height1 / width1)), interpolation=cv2.INTER_NEAREST)
        scale = height1 / image.shape[0]
        border_top = (input_size - image.shape[0]) // 2
        image = cv2.copyMakeBorder(image, border_top, input_size - image.shape[0] - border_top, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        input_data = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), axis=0)

        # Normalize if using float model
        if input_details[0]['dtype'] == np.float32:
            input_data = (np.float32(input_data) - 127.5) / 127.5

        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Perform inference
        t1 = time()
        interpreter.invoke()
        t2 = time()

        # Extract output and interpret results
        outname = output_details[0]['name']
        if 'StatefulPartitionedCall' in outname:  # TF2 model
            locations_idx, classes_idx, scores_idx, detections_idx = 1, 3, 0, 2
        else:  # TF1 model
            locations_idx, classes_idx, scores_idx, detections_idx = 0, 1, 2, 3

        locations = (interpreter.get_tensor(output_details[locations_idx]['index'])[0] * width1).astype(int)
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0].astype(int)
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]
        n_detections = int(interpreter.get_tensor(output_details[detections_idx]['index'])[0])

        # Draw the bounding boxes
        for i in range(n_detections):
            if scores[i] > MINIMUM_SCORE:
                y1, x1, y2, x2 = (locations[i, 0] - int(border_top * scale), locations[i, 1], locations[i, 2] - int(border_top * scale), locations[i, 3])
                draw_bounding_boxes(image_original, labels, x1, x2, y1, y2, classes[i])

        # Display inference time
        cv2.rectangle(image_original, (0, 0), (130, 20), (255, 0, 0), -1)
        cv2.putText(image_original, "inf time: %.3fs" % (t2 - t1), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Show the frame
        cv2.imshow("Object Detection", image_original)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
