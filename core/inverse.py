import torch


class InverseBatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        assert input.size(1) == input.size(2), "Input must be square matrices"
        inv_input = torch.inverse(input)
        ctx.save_for_backward(inv_input)
        return inv_input

    @staticmethod
    def backward(ctx, grad_output):
        (inv_input,) = ctx.saved_tensors  # [N, h, h]
        grad_input = -inv_input.transpose(1, 2).bmm(grad_output).bmm(inv_input)
        return grad_input


def InverseBatchFun(input: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse of a batch of square matrices.

    Args:
        input: Tensor of shape [N, h, h]

    Returns:
        H: Tensor of shape [N, h, h], batch of inverses
    """
    assert input.size(1) == input.size(2), "Input must be a batch of square matrices"
    return torch.linalg.inv(input)
