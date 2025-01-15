import torch
import torch.nn as nn

input = torch.Tensor(4, 8, 3, 128, 128)
input1 = input.permute(1, 0, 2, 3, 4)
print(input.shape)
print(input1.shape)
m = nn.Conv3d(4, 1, kernel_size=1, stride=1)
# m = nn.Conv3d(None, 1, 1)
m.in_channels = 4
out = m(input1)
print(out.shape)
out1 = out.squeeze(dim=1)
print(out1.shape)
