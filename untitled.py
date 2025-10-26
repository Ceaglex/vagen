import torch

# 检查是否有可用的GPU
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"当前系统中可用的GPU数量为：{device_count}")
else:
    print("未检测到可用的GPU")

# 使用GPU进行计算任务
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    z = x + y
    print(f"在GPU上进行计算：{z}")
else:
    print("无法使用GPU进行计算")
