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
CAPTURE_DEVICE = os.environ.get("CAPTURE_DEVICE", "/dev/video2")
CAPTURE_RESOLUTION_X = 1280
CAPTURE_RESOLUTION_Y = 960
CAPTURE_FRAMERATE = 30
INFERENCE_SIZE = 224
SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", 0.2))  # Default threshold of 0.2

# Load class labels for classification model
with open("labels_mobilenet_quant_v1_224.txt", "r") as file:
    class_labels = file.read().splitlines()

# Callback for processing frames
def on_frame(sink, appsrc, interpreter, input_details, output_details, input_size, last_time):
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
            input_data = np.expand_dims(img_resized, axis=0)

            # Normalize input using quantization parameters and cast to UINT8
            scale, zero_point = input_details[0]['quantization']
            input_data = (input_data / scale + zero_point).astype(np.uint8)

            # Perform inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            t1 = time()
            interpreter.invoke()
            t2 = time()

            # Get prediction scores
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]
            scale, zero_point = output_details[0]['quantization']
            output_data = (output_data - zero_point) * scale

            # Identify and display predictions above the threshold
            predictions = [(class_labels[i], output_data[i]) for i in range(len(output_data)) if output_data[i] > SCORE_THRESHOLD]
            predictions = sorted(predictions, key=lambda x: x[1], reverse=True)  # Sort by score

            # Display predictions on the frame
            y_offset = 30
            for class_name, score in predictions:
                cv2.putText(
                    img_array, f"{class_name}: {score:.2f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
                )
                y_offset += 30

            # Calculate FPS
            fps = 1 / (t2 - last_time[0])
            last_time[0] = t2

            # Display FPS on the frame
            cv2.putText(
                img_array, f"FPS: {fps:.2f}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
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

    # Setup TensorFlow Lite interpreter with optional delegate
    delegates = [Delegate("/usr/lib/libvx_delegate.so")] if USE_HW_ACCELERATED_INFERENCE else []
    interpreter = Interpreter(
        model_path="mobilenet_v1_1.0_224_quant.tflite",
        experimental_delegates=delegates
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    print("input_details\n", input_details, "\n")
    output_details = interpreter.get_output_details()
    print("output_details\n", output_details, "\n")
    input_size = INFERENCE_SIZE  # Set input size based on model's expected input

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
        input_details,
        output_details,
        input_size,
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
