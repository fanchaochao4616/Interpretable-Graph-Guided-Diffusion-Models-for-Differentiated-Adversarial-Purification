import torch

def compute_accuracy(images, labels, model, normalize, device):
    model = model.to(device)
    model.eval()
    # 初始化计数器
    correct_top1 = 0
    correct_top5 = 0

    images = images.to(device)
    labels = labels.to(device)

    # 禁用梯度计算以节省内存
    with torch.no_grad():
        # 前向传播
        outputs = model(normalize(images))  # 形状: [batch_size, num_classes]

        # 计算 Top-1 准确率
        _, predicted = torch.max(outputs, 1)  # 预测的类别
        correct_top1 += (predicted == labels).sum().item()

        # 计算 Top-5 准确率
        _, top5_pred = outputs.topk(5, dim=1, largest=True, sorted=True)  # Top-5 预测
        labels_expanded = labels.view(-1, 1).expand_as(top5_pred)  # 扩展标签
        correct_top5 += (top5_pred == labels_expanded).sum().item()

    return correct_top1, correct_top5