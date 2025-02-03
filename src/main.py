import sys, getopt
import numpy as np
from time import time
import os
import cv2
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

## Import tflite runtime
# import ai_edge_litert as tflite #Tensorflow_Lite
import ai_edge_litert as tflite
import tensorflow as tf

## Read Environment Variables
USE_HW_ACCELERATED_INFERENCE = True

## The system returns the variables as Strings, so it's necessary to convert them where we need the numeric value
if os.environ.get("USE_HW_ACCELERATED_INFERENCE") == "0":
    USE_HW_ACCELERATED_INFERENCE = False

MINIMUM_SCORE = float(os.environ.get("MINIMUM_SCORE", default = 0.55))

CAPTURE_DEVICE = os.environ.get("CAPTURE_DEVICE", default = "/dev/video2")

CAPTURE_RESOLUTION_X = int(os.environ.get("CAPTURE_RESOLUTION_X", default = 640))

CAPTURE_RESOLUTION_Y = int(os.environ.get("CAPTURE_RESOLUTION_Y", default = 480))

CAPTURE_FRAMERATE = int(os.environ.get("CAPTURE_FRAMERATE", default = 30))

STREAM_BITRATE = int(os.environ.get("STREAM_BITRATE", default = 2048))

## Media factory that runs inference
class InferenceDataFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, **properties):
        super(InferenceDataFactory, self).__init__(**properties)

        # Setup frame counter for timestamps
        self.number_frames = 0
        self.duration = (1.0 / CAPTURE_FRAMERATE) * Gst.SECOND  # duration of a frame in nanoseconds

        # Create opencv Video Capture
        self.cap = cv2.VideoCapture(f'v4l2src device={CAPTURE_DEVICE} extra-controls="controls,horizontal_flip=1,vertical_flip=1" ' \
                                    f'! video/x-raw,width={CAPTURE_RESOLUTION_X},height={CAPTURE_RESOLUTION_Y},framerate={CAPTURE_FRAMERATE}/1 ' \
                                    f'! videoconvert primaries-mode=fast n-threads=4 ' \
                                    f'! video/x-raw,format=BGR ' \
                                    f'! appsink', cv2.CAP_GSTREAMER)
        
        # Create factory launch string
        self.launch_string = f'appsrc name=source is-live=true format=GST_FORMAT_TIME ' \
                             f'! video/x-raw,format=BGR,width={CAPTURE_RESOLUTION_X},height={CAPTURE_RESOLUTION_Y},framerate={CAPTURE_FRAMERATE}/1 ' \
                             f'! videoconvert primaries-mode=fast n-threads=4 ' \
                             f'! video/x-raw,format=I420 ' \
                             f'! x264enc bitrate={STREAM_BITRATE} speed-preset=ultrafast tune=zerolatency threads=4 ' \
                             f'! rtph264pay config-interval=1 name=pay0 pt=96 '
        
        # Setup execution delegate, if empty, uses CPU
        if(USE_HW_ACCELERATED_INFERENCE):
            delegates = [tflite.load_delegate("/usr/lib/libvx_delegate.so")]
        else:
            delegates = []

        # Create the tensorflow-lite interpreter
        self.interpreter = tflite.interpreter.Interpreter(
            model_path="midas-tflite-v2-1-small-lite-v1.tflite",
            experimental_delegates=delegates)

        # Allocate tensors.
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size=self.input_details[0]['shape'][1]


    # Funtion to be ran for every frame that is requested for the stream
    def on_need_data(self, src, length):

        if self.cap.isOpened():
            # Read the image from the camera
            ret, image_original = self.cap.read()

            if ret:
                # Resize the image to the size required for inference
                img = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB) / 255.0

                img_resized = tf.image.resize(img, [224,224], method='bicubic', preserve_aspect_ratio=False)
                img_resized = tf.transpose(img_resized, [2, 0, 1])
                img_resized = tf.reshape(img_resized, [1, 3, 224, 224])
                tensor = tf.convert_to_tensor(img_resized, dtype=tf.float32)

                self.interpreter.set_tensor(self.input_details[0]['index'], tensor)

                # Execute the inference
                t1=time()
                self.interpreter.invoke()
                t2=time()

                output = self.interpreter.get_tensor(self.output_details[0]['index'])
                prediction = output.reshape(224, 224)

                depth_min = prediction.min()
                depth_max = prediction.max()
                img_out = (255 * (prediction - depth_min) / (depth_max - depth_min)).astype("uint8")
                cv2.imwrite(f"{t1}.png", img_out)


                # Create and setup buffer
                data = GLib.Bytes.new_take(img_out.tobytes())
                buf = Gst.Buffer.new_wrapped_bytes(data)
                buf.duration = self.duration
                timestamp = self.number_frames * self.duration
                buf.pts = buf.dts = int(timestamp)
                buf.offset = timestamp
                self.number_frames += 1

                # Emit buffer
                retval = src.emit('push-buffer', buf)
                if retval != Gst.FlowReturn.OK:
                    print(retval)

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)


class RtspServer(GstRtspServer.RTSPServer):
    def __init__(self, **properties):
        super(RtspServer, self).__init__(**properties)
        # Create factory
        self.factory = InferenceDataFactory()

        # Set the factory to shared so it supports multiple clients
        self.factory.set_shared(True)

        # Add to "inference" mount point. 
        # The stream will be available at rtsp://<board-ip>:8554/inference
        self.get_mount_points().add_factory("/inference", self.factory)
        self.attach(None)

def main():
    Gst.init(None)
    server = RtspServer()
    loop = GLib.MainLoop()
    loop.run()

        

if __name__ == "__main__":
    main()