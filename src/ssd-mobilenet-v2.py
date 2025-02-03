import gi
import os
import numpy as np
import cv2
from time import time
from ai_edge_litert.interpreter import Interpreter, Delegate

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Environment variables and configuration
USE_HW_ACCELERATED_INFERENCE = True if os.environ.get("USE_HW_ACCELERATED_INFERENCE", "1") == "1" else False
MINIMUM_SCORE = float(os.environ.get("MINIMUM_SCORE", 0.55))
CAPTURE_DEVICE = os.environ.get("CAPTURE_DEVICE", "/dev/video2")
CAPTURE_RESOLUTION_X = 1280  # Updated resolution
CAPTURE_RESOLUTION_Y = 960   # Updated resolution
CAPTURE_FRAMERATE = 30
INFERENCE_SIZE = 300  # Smaller size for faster inference

# Helper function to draw bounding boxes (optimized)
def draw_bounding_boxes(img, labels, x1, x2, y1, y2, object_class):
    box_colors = [
        (254, 153, 143), (253, 156, 104), (253, 157, 13), (252, 204, 26),
        (254, 254, 51), (178, 215, 50), (118, 200, 60), (30, 71, 87),
        (1, 48, 178), (59, 31, 183), (109, 1, 142), (129, 14, 64)
    ]
    color = box_colors[object_class % len(box_colors)]
    cv2.rectangle(img, (x2, y2), (x1, y1), color, 2)
    cv2.putText(
        img, labels[object_class], (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
    )

# Callback for processing frames
def on_frame(sink, appsrc, interpreter, labels, input_details, output_details, input_size, last_time,
             scale_detection, zero_point_detection, scale_box, zero_point_box, scale_class, zero_point_class, scale_score, zero_point_score):
    sample = sink.emit("pull-sample")
    if sample:
        buf = sample.get_buffer()
        result, map_info = buf.map(Gst.MapFlags.READ)
        if not result:
            return Gst.FlowReturn.ERROR

        try:
            # Convert buffer to numpy array
            img_array = np.frombuffer(map_info.data, dtype=np.uint8).reshape((CAPTURE_RESOLUTION_Y, CAPTURE_RESOLUTION_X, 3))
            img_array = img_array.copy()  # Create a writable copy for processing

            # Resize image for inference
            img_resized = cv2.resize(img_array, (input_size, input_size))
            input_data = np.expand_dims(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB), axis=0)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            t1 = time()
            interpreter.invoke()
            t2 = time()

            output_data = interpreter.get_tensor(output_details[0]["index"])[0]
            detections = ((output_data - zero_point_detection) * scale_detection).astype("uint8")

            output_data = interpreter.get_tensor(output_details[1]["index"])[0]
            locations = (output_data - zero_point_box) * scale_box

            output_data = interpreter.get_tensor(output_details[2]["index"])[0]
            classes = ((output_data - zero_point_class) * scale_class).astype("uint8")

            output_data = interpreter.get_tensor(output_details[3]["index"])[0]
            scores = (output_data - zero_point_score) * scale_score

            # Draw bounding boxes for detected objects
            for i in range(int(detections)):
                if scores[i] > MINIMUM_SCORE:
                    y1, x1, y2, x2 = (locations[i] * [img_array.shape[0], img_array.shape[1], img_array.shape[0], img_array.shape[1]]).astype(int)
                    draw_bounding_boxes(img_array, labels, x1, x2, y1, y2, classes[i])

            # Calculate and display FPS
            fps = 1 / (t2 - last_time[0]) if t2 - last_time[0] > 0 else 0
            last_time[0] = t2
            
            # Display inference time with dark lime green text and black outline
            cv2.putText(
                img_array, f"Inference time: {t2 - t1:.3f}s", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA  # Black outline
            )
            cv2.putText(
                img_array, f"Inference time: {t2 - t1:.3f}s", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 205, 50), 1, cv2.LINE_AA  # Dark lime green text
            )

            # Display FPS with dark lime green text and black outline
            cv2.putText(
                img_array, f"FPS: {fps:.2f}", (5, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA  # Black outline
            )
            cv2.putText(
                img_array, f"FPS: {fps:.2f}", (5, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 205, 50), 1, cv2.LINE_AA  # Dark lime green text
            )


            # Push processed frame to appsrc
            data = img_array.tobytes()
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

    # Load model and labels
    with open("labelmap.txt", "r") as file:
        labels = file.read().splitlines()

    # Setup TensorFlow Lite interpreter with optional delegate
    delegates = [Delegate("/usr/lib/libvx_delegate.so")] if USE_HW_ACCELERATED_INFERENCE else []
    interpreter = Interpreter(
        model_path="ssd-mobilenet-v2.tflite",
        experimental_delegates=delegates
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    print("input_details\n", input_details, "\n")
    output_details = interpreter.get_output_details()
    print("output_details\n", output_details, "\n")
    input_size = INFERENCE_SIZE  # Set input size for faster inference

    # Get and dequantize the output
    scale_detection, zero_point_detection = output_details[0]['quantization']
    scale_box, zero_point_box = output_details[1]['quantization']
    scale_class, zero_point_class = output_details[2]['quantization']
    scale_score, zero_point_score = output_details[3]['quantization']

    # Initialize last_time for FPS calculation
    last_time = [time()]

    # Setup GStreamer pipeline for capture
    pipeline = Gst.parse_launch(
        f"v4l2src device={CAPTURE_DEVICE} ! "
        f"queue ! "
        f"video/x-raw,width={CAPTURE_RESOLUTION_X},height={CAPTURE_RESOLUTION_Y},framerate={CAPTURE_FRAMERATE}/1 ! "
        f"videoconvert ! "
        f"video/x-raw,format=BGR ! "
        f"appsink name=sink emit-signals=true max-buffers=1 drop=true"
    )

    # Setup GStreamer pipeline for output to waylandsink
    output_pipeline = Gst.parse_launch(
        "appsrc name=appsrc is-live=true format=GST_FORMAT_TIME ! "
        "queue ! "
        "videoconvert ! "
        "waylandsink sync=false"
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
        interpreter,
        labels,
        input_details,
        output_details,
        input_size,
        last_time,
        scale_detection, zero_point_detection, scale_box, zero_point_box, scale_class, zero_point_class, scale_score, zero_point_score
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