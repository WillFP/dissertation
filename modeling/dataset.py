import h5py
import torch


class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        with h5py.File(data_path, 'r') as f:
            self.boards = torch.tensor(f['boards'][:], dtype=torch.float32).permute(0, 3, 1, 2).contiguous()

            self.metadata = torch.tensor(f['metadata'][:], dtype=torch.float32).contiguous()

            evaluations = torch.tensor(f['evaluations'][:], dtype=torch.float32)
            if len(evaluations.shape) > 1 and evaluations.shape[1] == 1:
                evaluations = evaluations.squeeze(1)
            self.evaluations = evaluations.contiguous()

    def __len__(self):
        return len(self.evaluations)

    def __getitem__(self, idx):
        return self.boards[idx], self.metadata[idx], self.evaluations[idx]
