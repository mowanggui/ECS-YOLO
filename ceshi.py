import torch
import torch.nn as nn
import torch_pruning as tp
from torchvision.models import resnet18

from models.experimental import attempt_load
from models.yolo import Detect, BasicBlock_2, BasicBlock_1, BasicBlock_3, Snn_Conv2d, batch_norm_2d1, batch_norm_2d




# model = resnet18(pretrained=True).eval()
weights = "runs/train/exp44/weights/best.pt"
model = attempt_load(weights, map_location=torch.device('cuda:0'), fuse=False)
# print(model)
for p in model.parameters():
    p.requires_grad = True
ignored_layers = []
for m in model.modules():
    if isinstance(m, Detect):
        ignored_layers.append(m)

model.eval()
# 1. 构建依赖图
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=torch.randn(1, 3, 640, 640, device='cuda:0'))

# 2. 获取与model.conv1存在依赖的所有层，并指定需要剪枝的通道索引（此处我们剪枝第[2,6,9]个通道）
Groups = DG.get_all_groups(ignored_layers=[Detect])

# 3. 执行剪枝操作
for group in Groups:
    # idxs = [2, 4, 6]  # your pruning indices
    print("xxx")
    # group.prune(idxs=idxs)
    print(group)

# print(model)

# output = model(torch.randn(1, 3, 640, 640, device='cuda:0'))  # 尝试运行剪枝后的网络
