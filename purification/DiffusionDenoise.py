from utils import *
from purification import purify_imagenet, purify_cifar
def mask_purify(
    config,
    network_clf,
    model,
    diffusion,
    inputs,
):
    inputs = inputs.to(config.device)

    network_clf.eval().to(config.device)
    model.eval().to(config.device)

    if config.structure.dataset in ["ImageNet", "ImageNet-5k","ImageNet-Mini"]:
        purify_func=purify_imagenet
    elif config.structure.dataset in ["CIFAR10", ]:
        purify_func = purify_cifar

    mask, interpretations = generate_mask_and_interpretations(config, inputs, network_clf, model, diffusion, purify_func)

    x_pur = purify_func(
        config=config,
        model=model,
        images=inputs,
        diffusion=diffusion,
        interpretations=interpretations,
        mask=mask,
    )
    return x_pur
