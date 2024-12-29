import torch_pruning as tp
from torchvision.models import resnet18

from models.experimental import attempt_load
import torch
import os
from utils.datasets import create_dataloader
from utils.general import check_dataset

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 下面老是报错

weights = "runs/train/exp46/weights/best.pt"
model = attempt_load(weights, map_location=torch.device('cuda:0'), fuse=False)
# print(model)
# model.eval()
for p in model.parameters():
    p.requires_grad = True
ignored_layers = []
from models.yolo import Detect

# from models.common import ImplicitA, ImplicitM

for m in model.modules():
    if isinstance(m, Detect):
        ignored_layers.append(m)
# unwrapped_parameters = []
# for name, m in model.named_parameters():
#     if isinstance(m, (ImplicitA, ImplicitM,)):
#         unwrapped_parameters.append((name, 1))  # pruning 1st dimension of implicit matrix

print(ignored_layers)
# train_path = check_dataset("data/bdd100k.yaml")
example_inputs = torch.rand(1, 3, 640, 640, device='cuda:0')
# train_loader, dataset = create_dataloader(train_path, 640, 16 // 1, gs, single_cls,
#                                               hyp=hyp, augment=True, cache=opt.cache, rect=opt.rect, rank=LOCAL_RANK,
#                                               workers=workers, image_weights=opt.image_weights, quad=opt.quad,
#                                               prefix=colorstr('train: '), shuffle=True)
# imp = tp.importance.BNScaleImportance()
imp = tp.importance.MagnitudeImportance(p=1)  # L2 norm pruning
# Step 2.2. 初始化剪枝器
iterative_steps = 1  # progressive pruning
prune_rate = 0.5  # 剪枝率
pruner = tp.pruner.MagnitudePruner(model=model,
                                   example_inputs=example_inputs,
                                   importance=imp,
                                   iterative_steps=iterative_steps,
                                   pruning_ratio=prune_rate,
                                   ignored_layers=ignored_layers, )
# pruner = tp.pruner.BNScalePruner(model=model,
#                                    example_inputs=example_inputs,
#                                    importance=imp,
#                                    ignored_layers=ignored_layers,
#                                    iterative_steps=iterative_steps,
#                                    # global_pruning=True,
#                                    pruning_ratio=0.5,
#                                    round_to=8,
#                                    )
base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
for group in pruner.step(interactive=True):  # Warning: groups must be handled sequentially. Do not keep them as a list.
    print(group.details())  # 打印具体的group以及将要被剪枝的通道索引
    # 此处可以插入自定义代码，例如监控、打印、分析等
    group.prune()  # 交互地调用pruning完成剪枝
pruned_model = pruner.model
print(pruned_model)
# 统计剪枝后参数量
pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(pruned_model, example_inputs)
print(f"macs: {base_macs} -> {pruned_macs}")
print(f"nparams: {base_nparams} -> {pruned_nparams}")
macs_cutoff_ratio = (base_macs - pruned_macs) / base_macs
nparams_cutoff_ratio = (base_nparams - pruned_nparams) / base_nparams
print(f"macs cutoff ratio: {macs_cutoff_ratio}")
print(f"nparams cutoff ratio: {nparams_cutoff_ratio}")
save_path = weights.replace(".pt", "_pruned_bn_0.3.pt")

torch.save({"model": pruned_model.module if hasattr(pruned_model, 'module') else pruned_model}, save_path)

# model = resnet18(pretrained=True)
#
# # Importance criteria
# example_inputs = torch.randn(1, 3, 224, 224)
# imp = tp.importance.TaylorImportance()
#
# ignored_layers = []
# for m in model.modules():
#     if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
#         ignored_layers.append(m)  # DO NOT prune the final classifier!
#
# iterative_steps = 5  # progressive pruning
# pruner = tp.pruner.MagnitudePruner(
#     model,
#     example_inputs,
#     importance=imp,
#     iterative_steps=iterative_steps,
#     ch_sparsity=0.5,  # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
#     ignored_layers=ignored_layers,
# )
#
# base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
# for i in range(iterative_steps):
#     if isinstance(imp, tp.importance.TaylorImportance):
#         # Taylor expansion requires gradients for importance estimation
#         loss = model(example_inputs).sum()  # a dummy loss for TaylorImportance
#         loss.backward()  # before pruner.step()
#     pruner.step()
#     macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
#     # finetune your model here
#     # finetune(model)
# print(pruner.model)
