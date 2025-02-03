import gi
import os
import numpy as np
import cv2
from time import time
from ai_edge_litert.interpreter import Interpreter, Delegate

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Environment variables and configuration
USE_HW_ACCELERATED_INFERENCE = True if os.environ.get("USE_HW_ACCELERATED_INFERENCE", "0") == "1" else False
MINIMUM_SCORE = float(os.environ.get("MINIMUM_SCORE", 0.55))
CAPTURE_DEVICE = os.environ.get("CAPTURE_DEVICE", "/dev/video2")
CAPTURE_RESOLUTION_X = 1280
CAPTURE_RESOLUTION_Y = 960
DISPLAY_RESOLUTION_X = 1024
DISPLAY_RESOLUTION_Y = 600
CAPTURE_FRAMERATE = 30
INFERENCE_SIZE_OBJECT_DETECTION = 300
INFERENCE_SIZE_DEPTH_ESTIMATION = (224, 224)
GST_SRC  = "libcamerasrc"
GST_SINK = "autovideosink"
colormap = cv2.COLORMAP_VIRIDIS

# Predefine constants for normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Helper function to draw bounding boxes
def draw_bounding_boxes(img, labels, x1, x2, y1, y2, object_class):
    box_colors = [
        (254, 153, 143), (253, 156, 104), (253, 157, 13), (252, 204, 26),
        (254, 254, 51), (178, 215, 50), (118, 200, 60), (30, 71, 87),
        (1, 48, 178), (59, 31, 183), (109, 1, 142), (129, 14, 64)
    ]
    color = box_colors[object_class % len(box_colors)]
    cv2.rectangle(img, (x2, y2), (x1, y1), color, 2)
    cv2.putText(
        img, labels[object_class], (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3, cv2.LINE_AA
    )

# Preprocess function for depth estimation
def preprocess_depth(image, target_size=INFERENCE_SIZE_DEPTH_ESTIMATION, mean=MEAN, std=STD):
    image_resized = cv2.resize(image, target_size)
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_normalized = (image_normalized - mean) / std
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    return image_transposed

# Depth to RGB conversion
def depth_to_rgb(depth_array):
    depth_array = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
    depth_array = depth_array.astype(np.uint8)
    depth_colored = cv2.applyColorMap(255 - depth_array, colormap)
    return depth_colored

# Callback for processing frames
def on_frame(sink, appsrc, od_interpreter, de_interpreter, labels, od_input_details, od_output_details, de_input_details, de_output_details, od_input_size, de_input_size, od_scale, od_zero_point, de_scale, de_zero_point, last_time):
    sample = sink.emit("pull-sample")
    if sample:
        buf = sample.get_buffer()
        result, map_info = buf.map(Gst.MapFlags.READ)
        if not result:
            return Gst.FlowReturn.ERROR

        try:
            # Convert buffer to numpy array and create a writable copy
            img_array = np.frombuffer(map_info.data, dtype=np.uint8).reshape((CAPTURE_RESOLUTION_Y, CAPTURE_RESOLUTION_X, 3)).copy()
            img_array = cv2.rotate(img_array, cv2.ROTATE_180)

            # Preprocess for object detection
            img_resized_od = cv2.resize(img_array, (od_input_size, od_input_size))
            input_data_od = np.expand_dims(cv2.cvtColor(img_resized_od, cv2.COLOR_BGR2RGB), axis=0)
            if od_input_details[0]['dtype'] == np.float32:
                input_data_od = (np.float32(input_data_od) - 127.5) / 127.5
            # Perform object detection inference
            od_interpreter.set_tensor(od_input_details[0]['index'], input_data_od)

            # Preprocess for depth estimation
            preprocess_image_de = preprocess_depth(img_array)
            input_data_de = np.clip((preprocess_image_de / de_scale + de_zero_point), 0, 255).astype(np.uint8)
            input_data_de = np.expand_dims(input_data_de, axis=0)

            # Perform depth estimation inference
            de_interpreter.set_tensor(de_input_details[0]['index'], input_data_de)

            t1 = time()
            od_interpreter.invoke()
            de_interpreter.invoke()
            t2 = time()

            # Process object detection results
            locations = od_interpreter.get_tensor(od_output_details[0]['index'])[0]
            classes = od_interpreter.get_tensor(od_output_details[1]['index'])[0].astype(int)
            scores = od_interpreter.get_tensor(od_output_details[2]['index'])[0]
            detections = od_interpreter.get_tensor(od_output_details[3]['index'])[0]

            # Get and dequantize depth estimation output
            output_data_de = de_interpreter.get_tensor(de_output_details[0]['index']).squeeze()
            output_image_de = depth_to_rgb(output_data_de)
            output_resized_de = cv2.resize(output_image_de, (CAPTURE_RESOLUTION_X, CAPTURE_RESOLUTION_Y), interpolation=cv2.INTER_LINEAR)

            # Draw bounding boxes on the depth map
            for i in range(int(detections)):
                if scores[i] > MINIMUM_SCORE:
                    y1, x1, y2, x2 = (locations[i] * [CAPTURE_RESOLUTION_Y, CAPTURE_RESOLUTION_X, CAPTURE_RESOLUTION_Y, CAPTURE_RESOLUTION_X]).astype(int)
                    draw_bounding_boxes(output_resized_de, labels, x1, x2, y1, y2, classes[i])

            # Resize the original image to maintain aspect ratio and fit within 1/3 of the original size
            max_width = DISPLAY_RESOLUTION_X // 2
            max_height = DISPLAY_RESOLUTION_Y // 2
            aspect_ratio = img_array.shape[1] / img_array.shape[0]
            if img_array.shape[1] > img_array.shape[0]:
                small_width = max_width
                small_height = int(small_width / aspect_ratio)
            else:
                small_height = max_height
                small_width = int(small_height * aspect_ratio)
            small_img = cv2.resize(img_array, (small_width, small_height))

            # Calculate the position to place the small image on the top left corner
            x_offset = 0
            y_offset = 0

            # Overlay the small image onto the depth map
            output_resized_de[y_offset:y_offset + small_height, x_offset:x_offset + small_width] = small_img

            # Calculate and display FPS
            fps = 1 / (t2 - last_time[0]) if t2 - last_time[0] > 0 else 0
            last_time[0] = t2
            print(f"fps: {fps:.2f}, Inference time: {t2 - t1:.3f}s")

            # Display inference time with dark lime green text and black outline
            cv2.putText(
                output_resized_de, f"Inference time: {t2 - t1:.3f}s", (5, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5, cv2.LINE_AA  # Black outline
            )
            cv2.putText(
                output_resized_de, f"Inference time: {t2 - t1:.3f}s", (5, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 205, 50), 3, cv2.LINE_AA  # Dark lime green text
            )

            # Display FPS with dark lime green text and black outline
            cv2.putText(
                output_resized_de, f"FPS: {fps:.2f}", (5, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5, cv2.LINE_AA  # Black outline
            )
            cv2.putText(
                output_resized_de, f"FPS: {fps:.2f}", (5, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 205, 50), 3, cv2.LINE_AA  # Dark lime green text
            )

            # Convert to BGR format for GStreamer output
            data = output_resized_de.tobytes()
            buffer = Gst.Buffer.new_allocate(None, len(data), None)
            buffer.fill(0, data)
            buffer.pts = buf.pts
            buffer.duration = buf.duration
            appsrc.emit("push-buffer", buffer)

        finally:
            buf.unmap(map_info)

        return Gst.FlowReturn.OK

    return Gst.FlowReturn.OK

# Main function to setup and run the pipeline
def main():
    Gst.init(None)

    # Load labels for object detection
    with open("labelmap.txt", "r") as file:
        labels = file.read().splitlines()

    # Setup TensorFlow Lite interpreters with optional delegate
    delegates = [Delegate("/usr/lib/libvx_delegate.so")] if USE_HW_ACCELERATED_INFERENCE else []

    # Object detection interpreter
    od_interpreter = Interpreter(
        model_path="lite-model_ssd_mobilenet_v1_1_metadata_2.tflite",
        experimental_delegates=delegates
    )
    od_interpreter.allocate_tensors()
    od_input_details = od_interpreter.get_input_details()
    od_output_details = od_interpreter.get_output_details()

    # Depth estimation interpreter
    de_interpreter = Interpreter(
        model_path="fastdepth.tflite",
        experimental_delegates=delegates
    )
    de_interpreter.allocate_tensors()
    de_input_details = de_interpreter.get_input_details()
    de_output_details = de_interpreter.get_output_details()

    # Initialize last_time for FPS calculation
    last_time = [time()]

    # Setup GStreamer pipeline for capture
    pipeline = Gst.parse_launch(
        f"{GST_SRC} ! "
        f"queue ! "
        f"video/x-raw,width={CAPTURE_RESOLUTION_X},height={CAPTURE_RESOLUTION_Y},framerate={CAPTURE_FRAMERATE}/1 ! "
        f"videoconvert ! "
        f"videoflip method=rotate-180 ! " 
        f"video/x-raw,format=BGR ! "
        f"appsink name=sink emit-signals=true max-buffers=1 drop=true"
    )

    # Setup GStreamer pipeline for output to waylandsink
    output_pipeline = Gst.parse_launch(
        "appsrc name=appsrc is-live=true format=GST_FORMAT_TIME ! "
        "queue ! "
        "videoconvert ! "
        f"videoscale ! video/x-raw,width={DISPLAY_RESOLUTION_X},height={DISPLAY_RESOLUTION_Y} ! "
        f"{GST_SINK} sync=false"
    )

    # Configure appsrc properties
    appsrc = output_pipeline.get_by_name("appsrc")
    caps_str = f"video/x-raw,format=BGR,width={CAPTURE_RESOLUTION_X},height={CAPTURE_RESOLUTION_Y},framerate={CAPTURE_FRAMERATE}/1"
    appsrc.set_property("caps", Gst.Caps.from_string(caps_str))
    appsrc.set_property("format", Gst.Format.TIME)

    # Start the output pipeline
    output_pipeline.set_state(Gst.State.PLAYING)

    # Get the appsink element for pulling frames
    sink = pipeline.get_by_name("sink")
    sink.connect(
        "new-sample",
        on_frame,
        appsrc,
        od_interpreter,
        de_interpreter,
        labels,
        od_input_details,
        od_output_details,
        de_input_details,
        de_output_details,
        INFERENCE_SIZE_OBJECT_DETECTION,
        INFERENCE_SIZE_DEPTH_ESTIMATION,
        od_input_details[0]['quantization'][0],
        od_input_details[0]['quantization'][1],
        de_input_details[0]['quantization'][0],
        de_input_details[0]['quantization'][1],
        last_time
    )

    # Start the capture pipeline
    pipeline.set_state(Gst.State.PLAYING)

    # Create a GLib MainLoop
    loop = GLib.MainLoop()

    try:
        print("Pipeline running. Press Ctrl+C to exit.")
        loop.run()
    except KeyboardInterrupt:
        print("Exiting pipeline.")
    finally:
        pipeline.set_state(Gst.State.NULL)
        output_pipeline.set_state(Gst.State.NULL)
        print("Pipeline stopped.")

if __name__ == "__main__":
    main()