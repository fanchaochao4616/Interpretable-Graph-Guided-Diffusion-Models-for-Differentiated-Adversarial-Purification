import torch
from fastai.vision.all import Normalize

#半色调方法
normer = Normalize.from_stats(0.5, 0.5)
class DifferentiableDither(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, dither_matrix):
        binary_output = (image > dither_matrix).float()
        ctx.save_for_backward(image, dither_matrix)
        return binary_output

    @staticmethod
    def backward(ctx, grad_output):
        image, dither_matrix = ctx.saved_tensors

        temperature = 10.0  # 控制梯度陡峭程度
        diff = image - dither_matrix
        sigmoid_approx = torch.sigmoid(diff * temperature)
        grad_approx = temperature * sigmoid_approx * (1 - sigmoid_approx)

        return grad_output * grad_approx, None

def generate_bayer_matrix(n):
    matrix = torch.tensor([[0.0]])
    for _ in range(n):
        size = matrix.shape[0]
        new_size = size * 2
        new_matrix = torch.zeros((new_size, new_size), dtype=matrix.dtype)
        new_matrix[:size, :size] = 4 * matrix
        new_matrix[:size, size:] = 4 * matrix + 2
        new_matrix[size:, :size] = 4 * matrix + 3
        new_matrix[size:, size:] = 4 * matrix + 1
        matrix = new_matrix
    max_val = 4 ** n - 1
    matrix = matrix / max_val
    return matrix

def ordered_dithering(image, n=3, norm=True):
    if norm: image = normer.decodes(image)
    b, c, h, w = image.shape
    dither_matrix = generate_bayer_matrix(n)
    m = dither_matrix.shape[0]

    repeat_h = (h + m - 1) // m
    repeat_w = (w + m - 1) // m
    tiled_dither = dither_matrix.repeat(repeat_h, repeat_w)

    tiled_dither = tiled_dither[:h, :w].to(image.device).type_as(image)

    tiled_dither = tiled_dither.view(1, 1, h, w)

    halftone = DifferentiableDither.apply(image, tiled_dither)
    if norm: halftone = normer.encodes(halftone)
    return halftone
