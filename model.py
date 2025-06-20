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

# 抗锯齿处理 
mask_float = output_data[0, 0]   # 1024x1024, 值在0~1之间

resized_mask_float = cv2.resize(mask_float, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
blurred_mask = cv2.GaussianBlur(resized_mask_float, (5, 5), 0)
binary_mask = np.zeros_like(blurred_mask, dtype=np.uint8)
binary_mask[blurred_mask > 0.45] = 255 
kernel = np.ones((3, 3), np.uint8)
smoothed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

# 创建带透明通道的结果
cv2.imwrite("mask.png", smoothed_mask)
rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
rgba[:, :, 3] = smoothed_mask
cv2.imwrite("output.png", rgba)
foreground = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
foreground[:, :, :3] = image
foreground[:, :, 3] = smoothed_mask
cv2.imwrite("foreground.png", foreground)