import onnx
import onnxruntime
import torch.onnx
from model_def import ECNet
import json
import torch as t
from tools import generate_laplacian_pyram
import numpy as np
from numpy import random as rd


onnx_params = json.load(open("conf.json", "r", encoding="utf-8"))["convert_to_onnx"]
laplacian_level_count = onnx_params["laplacian_level_count"]
layer_count_of_every_unet = onnx_params["layer_count_of_every_unet"]
first_layer_out_channels_of_every_unet = onnx_params["first_layer_out_channels_of_every_unet"]
pth_file_path = onnx_params["pth_file_path"]
use_iaff = onnx_params["use_iaff"]
dummy_input_image_size = onnx_params["dummy_input_image_size"]
onnx_output_path = onnx_params["onnx_output_path"]
dynamic_bhw = onnx_params["dynamic_bhw"]
input_names = ["in%d" % (i,) for i in range(1, 1 + laplacian_level_count)]
# output_names = ["out%d" % (i,) for i in range(1, 1 + laplacian_level_count)]
output_names = ["output"]
iaff_r = onnx_params["iaff_r"]
use_psa = onnx_params["use_psa"]

dummy_inputs = rd.randint(0, 255, dummy_input_image_size[2:] + [dummy_input_image_size[1]]).astype(np.uint8)
batch_size = dummy_input_image_size[0]
dummy_inputs = [t.cat([t.from_numpy(np.expand_dims(np.transpose(i, axes=[2, 0, 1]), axis=0).astype(np.float32)).type(t.FloatTensor)] * batch_size, dim=0) for i in generate_laplacian_pyram(dummy_inputs, laplacian_level_count)[0]]
dummy_input_shapes = dict(list(zip(input_names, [list(i.size()) for i in dummy_inputs])))
print("dummy_inputs_shapes:", dummy_input_shapes)
model = ECNet(laplacian_level_count, layer_count_of_every_unet, first_layer_out_channels_of_every_unet, use_iaff, iaff_r, use_psa)
model.load_state_dict(t.load(pth_file_path))
model.eval()


names = input_names + output_names
print("input_names:", input_names)
print("output_names:", output_names)
if dynamic_bhw:
    dynamic_axes = {name: {0: "batch", 2: "height", 3: "width"} for name in names}
    print("dynamic_axes:", dynamic_axes)
    torch.onnx.export(model, dummy_inputs, onnx_output_path, input_names=input_names, output_names=output_names, opset_version=12, dynamic_axes=dynamic_axes)
else:
    torch.onnx.export(model, dummy_inputs, onnx_output_path, input_names=input_names, output_names=output_names, opset_version=12)