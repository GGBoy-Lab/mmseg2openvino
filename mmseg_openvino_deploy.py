import cv2
import numpy as np
from openvino.inference_engine import IECore

# 参数
IMG_FPATH = "./input/27-7.jpg"
IR_MODEL_FPATH = "./model/CAP_0522.xml"
DEVICE = "CPU"
num_classes = 3  # 类别数

# 加载图像
img = cv2.imread(IMG_FPATH)
if img is None:
    raise Exception('Image not found!')

# 加载IR模型
ie = IECore()
net = ie.read_network(model=IR_MODEL_FPATH, weights=IR_MODEL_FPATH[:-3] + "bin")
exec_net = ie.load_network(network=net, device_name=DEVICE)

# 归一化方法
mean = np.array([0.1570598, 0.14917149, 0.14580465])
std = np.array([0.13731193, 0.12907335, 0.12485505])

# 模型的输入数据准备
input_blob = next(iter(net.input_info))
input_shape = net.input_info[input_blob].input_data.shape


def preprocess_image(img, mean, std):
    resized_img = cv2.resize(img, (input_shape[3], input_shape[2]))
    preprocessed_img = np.transpose(resized_img, (2, 0, 1)).astype(np.float32) / 255.0

    for i in range(3):
        preprocessed_img[i] = (preprocessed_img[i] - mean[i]) / std[i]

    return np.expand_dims(preprocessed_img, axis=0)


preprocessed_img = preprocess_image(img, mean, std)

# 执行推理
outputs = exec_net.infer(inputs={input_blob: preprocessed_img})
# 获取输出数据
output_blob = next(iter(net.outputs))
output_data = outputs[output_blob].reshape((input_shape[2], input_shape[3]))

# 获取每一个类别的像素值
classIds = output_data.astype(np.uint8)

# 自定义mask
COLORS = np.array([[0, 0, 0],  # Background
                   [0, 200, 200],  # Class 1
                   [0, 0, 200]])  # Class 2

# 处理输出数据
segment_map = COLORS[classIds]
segment_map = cv2.resize(segment_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

# Mix original with segmentation map
img_with_segmentation = (0.8 * segment_map + 0.2 * img).astype(np.uint8)

result_image = img_with_segmentation
h = 512
w = 512
resampled_img = cv2.resize(result_image, (w, h), interpolation=cv2.INTER_NEAREST)


# 可视化结果
def show_img(img):
    cv2.imshow("Segmentation Result", img)
    cv2.imshow('result', resampled_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# show_img(img_with_segmentation)

second_class_mask = (segment_map == [0, 0, 200]).all(axis=2)
contours, _ = cv2.findContours(second_class_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    long_axis = max(w, h)
    short_axis = min(w, h)
    area = long_axis * short_axis
    print(area)
    print(f"Class 2 bounding box: ({x}, {y}) - ({x + w}, {y + h}), Long Axis: {long_axis}, Short Axis: {short_axis}")

    # 绘制边界框
    cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 显示带有边界框的结果
show_img(result_image)
