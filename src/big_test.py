from ai_edge_litert.interpreter import Interpreter
from ai_edge_litert.interpreter import Delegate

model_list = [
    'MiDaS_small_pt2e.tflite',
    'MiDaS_small_opt_default.tflite',
    'lite-model_ssd_mobilenet_v1_1_metadata_2.tflite',
    'midas-tflite-v2-1-small-lite-v1.tflite',
    'mobilenet_v1_1.0_224_quant.tflite',
]

for model_path in model_list:
    # load model
    print(model_path)

    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates = [Delegate("/usr/lib/libvx_delegate.so")]
            )
        interpreter.allocate_tensors()

    except Exception as e:
        print(e)