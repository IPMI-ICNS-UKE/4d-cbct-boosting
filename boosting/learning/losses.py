import torch
import torch.nn as nn
import torch.nn.functional as F


def _calculate_gradient_magnitude(
    image: torch.tensor, eps: float = 1e-6
) -> torch.tensor:
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
        loss += gradient_attention * _gradient_attention_loss(
            input=input, target=target
        )

    return loss


class L2Loss(nn.Module):
    def __init__(self, gradient_attention: float = 0.0):
        self.gradient_attention = gradient_attention

        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return _l2_loss(input, target, gradient_attention=self.gradient_attention)


class BoostingLoss(nn.Module):
    def __init__(self, dt_regularization: float = 0.0, gradient_attention: float = 0.0):
        self.dt_regularization = dt_regularization
        self.gradient_attention = gradient_attention

        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.dt_regularization > 0.0:
            # calculate difference between phase images
            dt = torch.diff(input, dim=0)
            loss += self.dt_regularization * dt.abs().mean()

        loss += _l2_loss(input, target, gradient_attention=self.gradient_attention)

        return loss
