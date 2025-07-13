from attacks import *
from utils import *
from networks.wrn_cifar10 import WideResNet
import torchvision

current_config = load_config('/home/fcc/codefile/HGDM/configs/ImageNet.yml')
transform_raw_to_clf = raw_to_clf(current_config.structure.dataset)
transform_clf_to_raw = clf_to_raw(current_config.structure.dataset)
transform_raw_to_diff = raw_to_diff(current_config.structure.dataset)
transform_diff_to_raw = diff_to_raw(current_config.structure.dataset)

model, diffusion = get_imagenet_ddpm(conf=current_config)

testLoader = import_data(
    dataset=current_config.structure.dataset,
    train=False,
    shuffle=False,
    bsize=25,
)

network_clf = torchvision.models.resnet50(pretrained=True).to("cuda").eval()

top1_acc=0
top5_acc=0
sum=0

for i, (images, labels) in enumerate(testLoader):
    if i == 5: break
    correct_top1, correct_top5 = compute_accuracy(images, labels, network_clf, transform_raw_to_clf,
                                                  current_config.device)
    sum = sum + len(labels)
    top1_acc = top1_acc + correct_top1
    top5_acc = top5_acc + correct_top5
print(f'top1_acc:{top1_acc / sum}, top5_acc:{top5_acc / sum}')

top1_acc = 0
top5_acc = 0
sum = 0
for i,(images,labels) in enumerate(testLoader):
    if i ==5:break
    x_pur = purify_imagenet(current_config, model, images, diffusion, None, None)
    correct_top1, correct_top5 = compute_accuracy(x_pur, labels, network_clf, transform_raw_to_clf, current_config.device)
    sum = sum + len(labels)
    top1_acc=top1_acc+correct_top1
    top5_acc=top5_acc+correct_top5
print(f'top1_acc:{top1_acc/sum}, top5_acc:{top5_acc/sum}')
