import einops
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from diverse_vit.models import DiverseViT, Recorder


def input_gradient_loss(y_hat, inputs, outputs, n_heads, normalize_grads='none'):
    def mask_head(output_grad, head_idx):
        # We want to isolate the gradient contribution of the `head_idx` head,
        # so we prevent the gradient flow through the other heads
        B, B, D = output_grad.shape
        hid_dim = D // n_heads

        output_grad[:, :, :(head_idx * hid_dim)] = 0
        output_grad[:, :, ((head_idx + 1) * hid_dim):] = 0
        return output_grad

    input = inputs[-1]
    output = outputs[-1]
    y_hat_max = torch.max(y_hat, dim=-1)[0]

    # Compute input gradient for each head independently
    input_gradients = []
    for head_idx in range(n_heads):
        handle = output.register_hook(lambda grad: mask_head(grad, head_idx))
        input_grad = torch.autograd.grad(y_hat_max.sum(), input, create_graph=True)[0]
        if normalize_grads == 'per_token':
            input_grad = F.normalize(input_grad, dim=2)
        input_gradients.append(input_grad)
        handle.remove()

    input_gradients = torch.stack(input_gradients)
    input_gradients = rearrange(input_gradients, 'h b n d -> b n h d')
    # we compute the cosine similarity between all head pairs for each token
    # we average over all tokens
    # we compute the square of similarities as we want to make them close to 0,
    # meaning independent heads
    per_token_sim = torch.matmul(input_gradients, input_gradients.permute(0, 1, 3, 2))
    per_token_sim = per_token_sim**2
    similarities = per_token_sim.mean(1)
    mask = torch.eye(n_heads).repeat(similarities.shape[0], 1, 1).bool().to(similarities.device)
    similarities *= ~mask
    loss_ig = similarities.mean()
    return loss_ig
