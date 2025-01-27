import torch

from preprocess import fen_to_tensor

evaluation_model_path = "models/best_eval_model.pt"
autoencoder_model_path = "models/best_autoencoder.pt"

from modeling.autoencoder.predict import load_model as load_autoencoder
from modeling.evaluation.predict import load_model as load_evaluator, predict_position


def predict_fen(fen):
    autoencoder = load_autoencoder(autoencoder_model_path, latent_dim=128, device='cpu')
    evaluator = load_evaluator(evaluation_model_path, latent_dim=128, device='cpu')

    board, metadata = fen_to_tensor(fen)
    evaluation = predict_position(
        evaluator,
        autoencoder,
        torch.FloatTensor(board).permute(2, 0, 1),
        torch.FloatTensor(metadata),
        device='cpu'
    )
    score = evaluation.item()

    return score
