"""Dataset module for microscopy cell segmentation.
Provides a CellDataset class that loads images and masks, applies
preprocessing and augmentation, and returns PyTorch tensors.

Usage:
    from data_utils.dataset import CellDataset
    ds = CellDataset(root_dir, preprocess_mode='basic', aug_strength='standard')
"""
import os
import re
import cv2
import numpy as np
import tifffile as tiff
import albumentations as A
import torch
from torch.utils.data import Dataset


def _make_augmentation(strength, target_size=None):
    transforms = []
    if strength == 'light':
        transforms.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])
    elif strength == 'standard':
        transforms.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                scale=(0.85, 1.15),
                rotate=(-45, 45),
                border_mode=cv2.BORDER_REFLECT,
                p=0.5,
            ),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
        ])
    elif strength is None or str(strength).lower() == 'none':
        transforms = []
    else:
        raise ValueError(f"Unknown augmentation strength: {strength!r}")

    # Always append resize if target_size provided so outputs are consistent
    if target_size is not None:
        transforms.append(A.Resize(height=target_size[0], width=target_size[1], interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST))

    return A.Compose(transforms) if transforms else None


class CellDataset(Dataset):
    """PyTorch Dataset for cell segmentation.

    Returns image tensor (1,H,W) float in [0,1] and mask tensor (H,W) long with labels {0,1,2}.
    """

    def __init__(self, root_dir, preprocess_mode='basic', aug_strength='standard', target_size=(256, 256), split_ids=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'Microscopy_images')
        alt1 = os.path.join(root_dir, 'Ground_truth_masks')
        alt2 = os.path.join(root_dir, 'Groud_truth_masks')
        self.mask_dir = alt1 if os.path.exists(alt1) else alt2
        self.preprocess_mode = preprocess_mode
        self.target_size = target_size
        self.aug = _make_augmentation(aug_strength, target_size=target_size)

        self.samples = self._find_samples(split_ids)
        if len(self.samples) == 0:
            raise RuntimeError('No samples found. Check paths and filenames.')

    def _find_samples(self, split_ids=None):
        pattern = re.compile(r'Microscope_image_Cell(\d+)\.tif')
        ids = []
        split_set = set(split_ids) if split_ids else None
        for f in sorted(os.listdir(self.img_dir)):
            m = pattern.match(f)
            if not m:
                continue
            cid = m.group(1)
            nuc = os.path.join(self.mask_dir, f'Nucleus_mask_Cell{cid}.tif')
            chrm = os.path.join(self.mask_dir, f'Chromocenter_mask_Cell{cid}.tif')
            if os.path.exists(nuc) and os.path.exists(chrm):
                if split_set is None or cid in split_set:
                    ids.append(cid)
        return ids

    def _preprocess_image(self, img_u16):
        img = img_u16.astype(np.float32)
        lo, hi = np.percentile(img, 1), np.percentile(img, 99.5)
        img = np.clip(img, lo, hi)
        img01 = (img - lo) / (hi - lo + 1e-8)

        if self.preprocess_mode == 'basic':
            result = img01
        elif self.preprocess_mode == 'full' or self.preprocess_mode == 'corrected':
            if self.preprocess_mode == 'full':
                img01 = cv2.bilateralFilter(img01, d=5, sigmaColor=0.08, sigmaSpace=3)
            k = 51
            if k % 2 == 0:
                k += 1
            k = max(k, 15)
            bg = cv2.GaussianBlur(img01, (k, k), 0)
            corr = img01 - bg
            lo2, hi2 = np.percentile(corr, 1), np.percentile(corr, 99.5)
            corr = np.clip(corr, lo2, hi2)
            result = (corr - lo2) / (hi2 - lo2 + 1e-8)
        else:
            raise ValueError('Unknown preprocess mode')

        return result.astype(np.float32)

    def _build_mask(self, nuc_raw, chrom_raw):
        nuc_fg = (nuc_raw == 0).astype(np.uint8)
        chrom_fg = (chrom_raw == 0).astype(np.uint8)
        mask = np.zeros_like(nuc_fg, dtype=np.uint8)
        mask[nuc_fg == 1] = 1
        mask[chrom_fg == 1] = 2
        return mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cid = self.samples[idx]
        img = tiff.imread(os.path.join(self.img_dir, f'Microscope_image_Cell{cid}.tif'))
        nuc = tiff.imread(os.path.join(self.mask_dir, f'Nucleus_mask_Cell{cid}.tif'))
        chrom = tiff.imread(os.path.join(self.mask_dir, f'Chromocenter_mask_Cell{cid}.tif'))

        img_pp = self._preprocess_image(img)
        mask = self._build_mask(nuc, chrom)

        # Apply augmentation pipeline (includes resize if target_size provided)
        if self.aug is not None:
            augmented = self.aug(image=img_pp, mask=mask)
            img_pp, mask = augmented['image'], augmented['mask']
        else:
            # If no augmentation pipeline, ensure resize to target_size
            if self.target_size is not None:
                img_pp = cv2.resize(img_pp, self.target_size, interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        img_t = torch.from_numpy(img_pp).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask.astype(np.int64))
        return img_t, mask_t
