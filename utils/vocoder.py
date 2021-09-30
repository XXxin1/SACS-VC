import torch

vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')


def vocoder_inverse(inp):
    return vocoder.inverse(inp).detach().cpu().numpy()[0]


def vocoder_inverse_batch(inp):
    return vocoder.inverse(inp)
