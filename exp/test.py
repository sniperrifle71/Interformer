import torch

# 创建一个3x3的张量
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

# 在第一个维度（行）上应用大小为2的滑动窗口
# 步长为1
result = x.unfold(1, size=2, step=1)

print(result)