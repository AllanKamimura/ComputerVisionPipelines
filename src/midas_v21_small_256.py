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
INFERENCE_SIZE = 256  # Model's expected input size

model_path = "midas_v21_small_256.tflite"

# Callback for processing frames
def on_frame(sink, appsrc, interpreter, input_details, output_details, input_size):
    sample = sink.emit("pull-sample")
    if sample:
        buf = sample.get_buffer()
        result, map_info = buf.map(Gst.MapFlags.READ)
        if not result:
            return Gst.FlowReturn.ERROR

        try:
            # Convert buffer to numpy array
            img_array = np.frombuffer(map_info.data, dtype=np.uint8).reshape((CAPTURE_RESOLUTION_Y, CAPTURE_RESOLUTION_X, 3))

            # Resize and normalize input image
            img_resized = cv2.resize(img_array, (192, input_size))
            input_data = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0  # Normalize to [0, 1] for float32 model

            # Perform inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            t1 = time()
            interpreter.invoke()
            t2 = time()

            # Extract output tensor (single-channel float image)
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]
            output_data = output_data.squeeze()  # Remove single-dimensional entries, shape is now (256, 256)

            # Normalize output for display as grayscale
            output_data = (output_data - output_data.min()) / (output_data.max() - output_data.min()) * 255.0
            output_data = output_data.astype(np.uint8)

            # Resize output back to original input size
            output_resized = cv2.resize(output_data, (CAPTURE_RESOLUTION_X, CAPTURE_RESOLUTION_Y), interpolation=cv2.INTER_LINEAR)

            # Display inference time on the output image
            cv2.putText(
                output_resized, f"Inference time: {t2 - t1:.3f}s", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA
            )

            # Convert to BGR format for GStreamer output
            output_rgb = cv2.cvtColor(output_resized, cv2.COLOR_GRAY2BGR)
            data = output_rgb.tobytes()
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
        model_path=model_path,  # Update to the new model path
        experimental_delegates=delegates
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    print("input_details\n", input_details, "\n")
    output_details = interpreter.get_output_details()
    print("output_details\n", output_details, "\n")
    input_size = INFERENCE_SIZE  # Set input size based on model's expected input

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
        input_size
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
