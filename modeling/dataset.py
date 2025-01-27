import h5py
import torch


class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path

        with h5py.File(h5_path, 'r') as f:
            self.length = len(f['boards'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            board = torch.FloatTensor(f['boards'][idx]).permute(2, 0, 1)
            metadata = torch.FloatTensor(f['metadata'][idx])
            evaluation = torch.tensor(f['evaluations'][idx], dtype=torch.float32)
            return board, metadata, evaluation
