import h5py
import torch


class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        with h5py.File(data_path, 'r') as f:
            self.boards = torch.tensor(f['boards'][:], dtype=torch.float32)
            self.metadata = torch.tensor(f['metadata'][:], dtype=torch.float32)
            self.evaluations = torch.tensor(f['evaluations'][:], dtype=torch.float32)

    def __len__(self):
        return len(self.evaluations)

    def __getitem__(self, idx):
        return self.boards[idx].permute(2, 0, 1), self.metadata[idx], self.evaluations[idx].squeeze()
