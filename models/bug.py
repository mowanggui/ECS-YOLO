import torch

path = 'D:/edge/Downloads/yolopv2.pt'

pretrained_dict = torch.load(path)

for k, v in pretrained_dict.named_parameters():  # k 参数名 v 对应参数值

    print(k,v)
