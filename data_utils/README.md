# data_utils README

This module provides a shared `CellDataset` for the training team.

Main configurable options

- `preprocess_mode`: `'basic'` or `'full'`
- `aug_strength`: `'none'`, `'light'`, or `'standard'`

Output format

The dataset returns:
- `image`: FloatTensor `(1, H, W)` normalized to `[0,1]`
- `mask`: LongTensor `(H, W)` with labels `{0, 1, 2}`

The model should output 3-class logits (do not apply softmax in the model). Use `torch.nn.CrossEntropyLoss()` directly with the returned mask tensor.

Validation / testing

For validation or testing, create the dataset with:

    aug_strength='none'

This disables random augmentations while still applying resizing.

Reproducibility

- Set random seeds (`numpy`, `torch`, `random`) in the training script.
- If using multiple `DataLoader` workers, define a `worker_init_fn` to ensure deterministic behavior.

Usage example

```python
from data_utils.dataset import CellDataset
from torch.utils.data import DataLoader

ROOT = '/content/drive/MyDrive/CMPUT469_Cell'  # adjust path
ds = CellDataset(ROOT, preprocess_mode='basic', aug_strength='standard', target_size=(256,256))
loader = DataLoader(ds, batch_size=4, shuffle=True)
images, masks = next(iter(loader))
```
