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
DISPLAY_RESOLUTION_X = 1024
DISPLAY_RESOLUTION_Y = 600
CAPTURE_FRAMERATE = 30
INFERENCE_SIZE = (256, 192)  # Model's expected input size

model_path = "midas-tflite-v2-1-small-lite-v1-quant.tflite"

# Predefine constants for normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(image, target_size=INFERENCE_SIZE, mean=MEAN, std=STD):
    """
    Preprocess the input image to match the TFLite model requirements.

    Args:
        image (np.ndarray): Input image in BGR format with shape (H, W, C).
        target_size (tuple): Target size (height, width) for resizing.
        mean (np.ndarray): Mean values for normalization.
        std (np.ndarray): Standard deviation values for normalization.

    Returns:
        np.ndarray: Preprocessed image with shape (192, 256, 3) and dtype float32.
    """
    # Resize the image to the target size
    image_resized = cv2.resize(image, INFERENCE_SIZE)
    
    # Normalize to [0, 1] by dividing by 255
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Apply mean and std normalization
    image_normalized = (image_normalized - mean) / std
    
    return image_normalized

# Callback for processing frames
def on_frame(sink, appsrc, interpreter, input_details, output_details, input_size, scale, zero_point, last_time):
    sample = sink.emit("pull-sample")
    if sample:
        buf = sample.get_buffer()
        result, map_info = buf.map(Gst.MapFlags.READ)
        if not result:
            return Gst.FlowReturn.ERROR

        try:
            # Convert buffer to numpy array
            img_array = np.frombuffer(map_info.data, dtype=np.uint8).reshape((CAPTURE_RESOLUTION_Y, CAPTURE_RESOLUTION_X, 3))

            preprocess_image = preprocess(img_array)
            input_data = np.clip((preprocess_image / scale + zero_point), 0, 255).astype(np.uint8)
            input_data = np.expand_dims(input_data, axis=0)  # Shape (1, 192, 256, 3)

            # Perform inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            t1 = time()
            interpreter.invoke()
            t2 = time()

            # Get and dequantize the output
            output_data = interpreter.get_tensor(output_details[0]['index']).squeeze()

            # Rotate or transpose the output to match the input format
            # Transpose (H, W) -> (W, H)
            output_image = np.transpose(output_data, (1, 0))


            # Resize output back to original input size
            output_resized = cv2.resize(output_image, (CAPTURE_RESOLUTION_X, CAPTURE_RESOLUTION_Y), interpolation=cv2.INTER_LINEAR)
            
            # Calculate and display FPS
            fps = 1 / (t2 - last_time[0]) if t2 - last_time[0] > 0 else 0
            last_time[0] = t2

            print(f"fps: {fps:.2f}, Inference time: {t2 - t1:.3f}s")
            # Display inference time with dark lime green text and black outline
            cv2.putText(
                output_resized, f"Inference time: {t2 - t1:.3f}s", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA  # Black outline
            )
            cv2.putText(
                output_resized, f"Inference time: {t2 - t1:.3f}s", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 205, 50), 1, cv2.LINE_AA  # Dark lime green text
            )

            # Display FPS with dark lime green text and black outline
            cv2.putText(
                output_resized, f"FPS: {fps:.2f}", (5, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA  # Black outline
            )
            cv2.putText(
                output_resized, f"FPS: {fps:.2f}", (5, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 205, 50), 1, cv2.LINE_AA  # Dark lime green text
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

    scale, zero_point = input_details[0]['quantization']

    # Initialize last_time for FPS calculation
    last_time = [time()]

    # Setup GStreamer pipeline for capture
    pipeline = Gst.parse_launch(
        f"libcamerasrc ! "
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
        f"videoscale ! video/x-raw,width={DISPLAY_RESOLUTION_X},height={DISPLAY_RESOLUTION_Y} ! "
        "autovideosink sync=false"
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
        INFERENCE_SIZE,
        scale, zero_point,
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
