import torch

from modeling.model import ChessEvaluator


def load_model(model_path, device='cpu'):
    model = ChessEvaluator()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model
