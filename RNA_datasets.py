import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class RNADataset(Dataset):
    def __init__(self, X, y):
        self.X = X.tocsr()
        self.y = np.asarray(y, dtype=np.int64)

        if self.X.shape[0] != len(self.y):
            raise ValueError(f"X rows ({self.X.shape[0]}) != y length ({len(self.y)})")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].toarray().ravel().astype(np.float32, copy=False)
        y = np.int64(self.y[idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
