import gi
import os
import numpy as np
import cv2
from time import time
import tensorflow as tf

options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Environment variables and configuration
USE_HW_ACCELERATED_INFERENCE = True if os.environ.get("USE_HW_ACCELERATED_INFERENCE", "1") == "1" else False
CAPTURE_DEVICE = os.environ.get("CAPTURE_DEVICE", "/dev/video0")
CAPTURE_RESOLUTION_X = 1280
CAPTURE_RESOLUTION_Y = 720
CAPTURE_FRAMERATE = 30
INFERENCE_SIZE = (256, 192)  # Model's expected input size

model_path = "midas_v21_small_256"

# Predefine constants for normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Assuming depth_array is your depth output of shape (224, 224)
def depth_to_rgb(depth_array):
    # Normalize depth values to 0-255
    depth_array = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
    depth_array = depth_array.astype(np.uint8)
    
    depth_colored = cv2.applyColorMap(depth_array, cv2.COLORMAP_VIRIDIS)
    return depth_colored

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
            img_array = cv2.cvtColor(img_array.copy(), cv2.COLOR_BGR2RGB)  # Create a writable copy for processing

            preprocess_image = preprocess(img_array)
            input_data = np.clip((preprocess_image / scale + zero_point), 0, 1).astype(np.float32)
            input_data = tf.convert_to_tensor(np.expand_dims(input_data, axis=0))  # Shape (1, 192, 256, 3)

            t1 = time()
            output = interpreter(input_data)
            t2 = time()

            output_name = list(output.keys())[0]
            output_data = output[output_name].numpy().squeeze()

            # Rotate or transpose the output to match the input format
            # Transpose (H, W) -> (W, H)
            # output_image = np.transpose(output_data, (1, 0))

            output_image = depth_to_rgb(output_data)

            # Resize output back to original input size
            output_resized = cv2.resize(output_image, (CAPTURE_RESOLUTION_X, CAPTURE_RESOLUTION_Y), interpolation=cv2.INTER_LINEAR)

            # Calculate and display FPS
            fps = 1 / (t2 - last_time[0]) if t2 - last_time[0] > 0 else 0
            last_time[0] = t2

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

            print(f"fps: {fps:.2f}, Inference time: {t2 - t1:.3f}s")
            # Convert to BGR format for GStreamer output
            # output_rgb = cv2.cvtColor(output_resized, cv2.COLOR_GRAY2BGR)
            
            data = output_resized.tobytes()
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
    model = tf.saved_model.load(model_path)

    # Get the signature for inference (adjust signature key if necessary)
    interpreter = model.signatures["serving_default"]

    input_details = interpreter.inputs[0]
    print("input_details\n", input_details, "\n")
    output_details = interpreter.outputs[0]
    print("output_details\n", output_details, "\n")

    scale, zero_point = (1, 0)


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
