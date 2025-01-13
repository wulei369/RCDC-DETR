import tensorrt as trt
import torch
from torch import nn
from torchvision import models  # 假设您使用的是torchvision中的模型作为示例
from torch.onnx import export
import numpy as np

# 假设您有一个加载模型的函数（这里以torchvision的模型为例）
def load_model():
    # 这里应该是加载您的RT-DETR模型的代码
    # 例如：return torch.load('best.pt', map_location=torch.device('cpu'))['model']
    # 但由于RT-DETR不是torchvision的一部分，这里我们用ResNet作为示例
    return models.resnet18(pretrained=True)  # 替换为您的RT-DETR模型加载代码

# 加载模型
model = 'best.pt'
model.eval()  # 设置为评估模式

# 设置ONNX导出参数
input_shape = (1, 3, 800, 800)  # 根据实际输入尺寸调整
dummy_input = torch.randn(input_shape)  # 用于指定输入形状的随机张量
output_model_path = "rt_detr.onnx"

# 导出模型为ONNX格式
export(model, dummy_input, output_model_path, verbose=False, input_names=['input'], output_names=['output'])

# 使用TensorRT转换ONNX模型为TensorRT引擎
logger = trt.Logger(trt.Logger.WARNING)  # 设置日志级别
with trt.Builder(logger) as builder, builder.create_network() as network, trt.OnnxParser(network, logger) as parser:
    builder.max_workspace_size = 1 << 30  # 设置最大工作空间大小（1GB）
    with open(output_model_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    engine = builder.build_cuda_engine(network)
    engine.save_to_file('rt_detr.trt')

# 使用TensorRT进行推理的简化示例
def infer_with_tensorrt(engine_path, input_data):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        # 分配输入和输出绑定内存
        inputs = [np.ascontiguousarray(input_data.cpu().numpy())]
        output = np.empty(tuple(engine.get_binding_shape(1)), dtype=np.float32)

        # 设置动态输入尺寸（如果需要的话）
        # profile = engine.get_profile(0)  # 如果有多个profile，选择需要的那个
        # profile.set_shape('input', (batch_size, 3, height, width))
        # context.set_binding_shape(0, (batch_size, 3, height, width))  # 第一个binding是输入

        # 执行推理
        [context.execute_async_v2(bindings=[inputs[0].ctypes.data, output.ctypes.data])]
        # 同步CUDA流（如果需要异步执行的话）
        # cuda.synchronize()

        # 处理输出（根据您的模型输出格式）
        # 例如：output_data = postprocess_output(output)
        return output

# 示例输入数据（需要替换为您的实际数据）
input_data = torch.randn(input_shape)

# 进行推理
output_data = infer_with_tensorrt('rt_detr.trt', input_data)
print(output_data)