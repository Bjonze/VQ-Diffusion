import os, glob, numpy as np, torch
from torch.utils.data import Dataset

class NPZIndicesCtxDataset(Dataset):
    def __init__(self, data_root, phase, max_len=512, dtype_idx=np.int64, dtype_ctx=np.float32):
        # root like: /data/Data/latent_vectors/vqgan/ema_8x8x8_ctx
        self.paths = sorted(glob.glob(os.path.join(data_root, phase, "*.npz")))
        assert len(self.paths) > 0, f"No npz files found in {data_root}/{phase}"
        self.max_len = max_len
        self.dtype_idx = dtype_idx
        self.dtype_ctx = dtype_ctx

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        arr = np.load(self.paths[i])
        idx = arr["indices"].astype(self.dtype_idx).reshape(-1)
        assert idx.shape[0] == self.max_len, f"Expected {self.max_len} tokens, got {idx.shape}"
        ctx = arr["ctx"].astype(self.dtype_ctx).reshape(-1)  # (18,)
        sample = {
            "indices": torch.from_numpy(idx),     # (512,)
            "ctx": torch.from_numpy(ctx).float(), # (18,)
        }
        return sample