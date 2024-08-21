import nncf
import torch
from torchvision import datasets, transforms
import openvino.runtime as ov

# 加载需要量化的模型
model = ov.Core().read_model("./IR/FP32.xml")

train_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomCrop((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.21390933, 0.20562113, 0.20215689], std=[0.2041261,  0.19696569, 0.19321205]),
])

# 校准集数据加载
# calibration_dataset = ImageFolder("./calibration_dataset", transform=train_transforms)

# 校准数据集预处理函数
val_dataset = datasets.ImageFolder("data", transform=train_transforms)
dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)


# 第一步 初始化 transformation 函数
def transform_fn(data_item):
    images, _ = data_item
    return images


# 第二步 初始化数据集
calibration_dataset = nncf.Dataset(dataset_loader, transform_fn)
# 第三步，推理nncf.quantize
quantized_model = nncf.quantize(model, calibration_dataset)
# 保存量化后的模型
ov.serialize(quantized_model, "./IR/INT8.xml")
