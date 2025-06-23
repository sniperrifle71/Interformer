import torch

# 假设 xlayerresults 包含两个形状为 (2, 3, 4) 的张量
xlayerresults = [
    torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                  [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]),
    torch.tensor([[[25, 26, 27, 28], [29, 30, 31, 32], [33, 34, 35, 36]],
                  [[37, 38, 39, 40], [41, 42, 43, 44], [45, 46, 47, 48]]])
]

# 第一行代码
x1 = torch.stack(xlayerresults, dim=2).sum(dim=2)

# 第二行代码
x2 = torch.stack(xlayerresults, dim=1).sum(dim=1)

print("x1:\n", x1)
print("x2:\n", x2)
