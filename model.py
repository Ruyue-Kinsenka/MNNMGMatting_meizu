import MNN
import numpy as np
import cv2

model_path = "model/meizu.mnn"
interpreter = MNN.Interpreter(model_path)
session = interpreter.createSession()

input_tensor = interpreter.getSessionInput(session)

image = cv2.imread("input.jpg")
orig_h, orig_w = image.shape[:2]

resized = cv2.resize(image, (1024, 1024))
normalized = resized.astype(np.float32) / 255.0
rgb_normalized = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)

# 创建输入Tensor (NHWC格式)
tmp_input = MNN.Tensor((1, 1024, 1024, 3), MNN.Halide_Type_Float,
                      rgb_normalized, MNN.Tensor_DimensionType_Tensorflow)

# 输入数据拷贝
input_tensor.copyFrom(tmp_input)
interpreter.runSession(session)

# 获取输出Tensor
output_tensor = interpreter.getSessionOutput(session)

# 创建输出缓冲区
output_shape = output_tensor.getShape()
output_buffer = MNN.Tensor(output_shape, MNN.Halide_Type_Float,
                          np.ones(output_shape).astype(np.float32),
                          MNN.Tensor_DimensionType_Caffe)

# 拷贝输出数据
output_tensor.copyToHostTensor(output_buffer)

# 获取numpy数组
output_data = output_buffer.getNumpyData()

# 处理输出
mask = output_data[0, 0]
mask = (mask > 0.5).astype(np.uint8) * 255
resized_mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

# 创建带透明通道的结果
cv2.imwrite("mask.png", resized_mask)
rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
rgba[:, :, 3] = resized_mask
cv2.imwrite("output.png", rgba)
foreground = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
foreground[:, :, :3] = image
foreground[:, :, 3] = resized_mask
cv2.imwrite("foreground.png", foreground)