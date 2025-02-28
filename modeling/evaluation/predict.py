import torch

from modeling.autoencoder import encode_position
from modeling.evaluation import ChessEvaluationCNN


def load_model(model_path, latent_dim=64, device='cpu'):
    """
    Load a trained (pure) autoencoder model from disk.

    Args:
        model_path (str): Path to the saved model checkpoint
        latent_dim (int): Dimension of the latent space
        device (str): Device to load the model on ('cpu' or 'cuda')

    Returns:
        model: Loaded ChessAutoencoder model
    """
    model = ChessEvaluationCNN(latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model


def predict_position(model, autoencoder, board, metadata, device='cpu'):
    model.eval()
    autoencoder.eval()
    with torch.inference_mode():
        board = board
        metadata = metadata
        latent = torch.FloatTensor(encode_position(autoencoder, board, metadata, device)).to(device)
        prediction = model(latent)
    return prediction.cpu()
