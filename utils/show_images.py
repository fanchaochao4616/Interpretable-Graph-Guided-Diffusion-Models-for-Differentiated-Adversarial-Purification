import matplotlib.pyplot as plt
def show_images(images, labels=None, denormalize=True):
    nrows=2
    ncols=int(images.shape[0]/2)
    # 创建画布
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 8))
    axes = axes.flatten()

    for i, (img, ax) in enumerate(zip(images, axes)):
        img = img.detach().cpu().numpy().transpose((1, 2, 0))
        ax.imshow(img)
        if labels is not None:
            ax.set_title(f"Label: {labels[i].item()}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()