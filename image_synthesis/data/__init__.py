from .npz_indices_ctx_dataset import NPZIndicesCtxDataset

def build_dataset(name, **kwargs):
    if name.lower() == "npz_indices_ctx":
        return NPZIndicesCtxDataset(**kwargs)
    # ... keep existing branches
    raise ValueError(f"Unknown dataset {name}")