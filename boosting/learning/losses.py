import torch
import torch.nn.functional as F
import torch.nn as nn


def _calculate_gradient_magnitude(image: torch.tensor, eps: float = 1e-6) -> torch.tensor:
    x_grad, y_grad = torch.gradient(image, dim=(-2, -1))
    return torch.sqrt((eps + x_grad**2 + y_grad**2))


def _gradient_attention_loss(input: torch.tensor, target: torch.tensor):
    input_grad = _calculate_gradient_magnitude(input)
    target_grad = _calculate_gradient_magnitude(target)

    return F.l1_loss(input_grad, target_grad)


def _l2_loss(
    input: torch.tensor, target: torch.tensor, gradient_attention: float = 0.0
):
    loss = F.mse_loss(input, target, reduction="mean")
    if gradient_attention > 0.0:
        loss += gradient_attention * _gradient_attention_loss(input=input, target=target)

    return loss


class L2Loss(nn.Module):
    def __init__(self, gradient_attention: float = 0.0):
        self.gradient_attention = gradient_attention

        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return _l2_loss(input, target, gradient_attention=self.gradient_attention)
