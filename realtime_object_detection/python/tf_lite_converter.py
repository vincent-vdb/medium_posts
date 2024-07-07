import argparse

import torch
import tensorflow as tf
import onnx
from onnx import helper
import onnx_tf

from blazeface import BlazeFace, ModelParameters


def fix_onnx_naming(onnx_model, model_name):
    # Define a mapping from old names to new names
    name_map = {"x.1": "x_1"}

    # Initialize a list to hold the new inputs
    new_inputs = []

    # Iterate over the inputs and change their names if needed
    for inp in onnx_model.graph.input:
        if inp.name in name_map:
            # Create a new ValueInfoProto with the new name
            new_inp = helper.make_tensor_value_info(name_map[inp.name],
                                                    inp.type.tensor_type.elem_type,
                                                    [dim.dim_value for dim in inp.type.tensor_type.shape.dim])
            new_inputs.append(new_inp)
        else:
            new_inputs.append(inp)

    # Clear the old inputs and add the new ones
    onnx_model.graph.ClearField("input")
    onnx_model.graph.input.extend(new_inputs)

    # Go through all nodes in the model and replace the old input name with the new one
    for node in onnx_model.graph.node:
        for i, input_name in enumerate(node.input):
            if input_name in name_map:
                node.input[i] = name_map[input_name]
    onnx.save(onnx_model, f'{model_name}.onnx')
    return onnx_model

def convert_torch_to_tflite(torch_path, model_name, model_params):

    if model_params.image_size == 256:
        model = BlazeFace(back_model=True)
    else:
        model = BlazeFace()
    model.load_anchors('anchors.npy')
    model.load_state_dict(torch.load(torch_path))
    model.eval()
    model.to('cpu')

    input_shape = (1, 3, model_params.image_size, model_params.image_size)

    torch.onnx.export(model, torch.randn(input_shape), f'{model_name}.onnx', opset_version=11)
    onnx_model = onnx.load(f'{model_name}.onnx')
    onnx_model = fix_onnx_naming(onnx_model, model_name)
    tf_model = onnx_tf.backend.prepare(onnx_model)
    tf_model.export_graph(f'{model_name}.tf')


    model_converter = tf.lite.TFLiteConverter.from_saved_model(f'{model_name}.tf')
    model_converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    # model_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = model_converter.convert()
    open(f'{model_name}.tflite', 'wb').write(tflite_model)


if __name__ == '__main__':
    # Parse the args
    parser = argparse.ArgumentParser(description='Convert BlazeFace torch model to TFLite format')
    parser.add_argument('--input_path', help='path to input torch model', type=str, default='weights/blazeface.pt')
    parser.add_argument('--output_name', help='output filename root', type=str, default='weights/blazeface')
    parser.add_argument('--input_img_size', help='model input image size', type=int, default=128)

    args = parser.parse_args()

    model_params = ModelParameters(image_size=args.input_img_size)
    convert_torch_to_tflite(args.input_path, args.output_name, model_params)
