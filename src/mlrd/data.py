"""NACC-only paired data; no clinical fields are read or transformed."""
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from paired_t1_fdg_dataset import PairedT1FDGDataset

FEATURES = []

def read_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding='utf-8'))
    if cfg['model'].get('clinical_dim', 8) != 0:
        raise ValueError('MRI-only NACC requires clinical_dim=0')
    for key in ('pairs_csv', 'split_csv', 'prepared_dir', 'cache_dir'):
        value = cfg['data'].get(key)
        if value and not Path(value).is_absolute():
            cfg['data'][key] = str((path.parent / value).resolve())
    cfg['output_dir'] = str((path.parent / cfg['output_dir']).resolve()) if not Path(cfg['output_dir']).is_absolute() else cfg['output_dir']
    return cfg

def prepare_metadata(pairs_path, split_path, clinical_path=None, repair_folds=False, max_clinical_days=180):
    pairs, splits = pd.read_csv(pairs_path), pd.read_csv(split_path)
    if pairs.duplicated(['subject_id', 'sample_id']).any() or splits.duplicated(['subject_id', 'sample_id']).any():
        raise ValueError('Duplicate NACC pair/split keys')
    df = pairs.merge(splits, on=['subject_id', 'sample_id'], how='inner', validate='one_to_one')
    if len(df) != len(splits) or df.groupby('subject_id').split.nunique().gt(1).any():
        raise ValueError('NACC split coverage or subject leakage error')
    df['clinical_status'] = 'not_used_mri_only'
    report = dict(pair_rows=len(pairs), split_rows=len(splits), paired_in_split=len(df),
                  clinical_features=[], conditioning='mri_only', clinical_source='not used by design')
    return df, splits, report


def select_rows(df, split, fold):
    if split == 'train':
        return df[df.split.eq('train') & df.fold.ne(fold)].copy()
    if split == 'val':
        return df[df.split.eq('train') & df.fold.eq(fold)].copy()
    return df[df.split.eq(split)].copy()


class ClinicalTransform:
    """Empty tensor preserves the trainer call signature, not a clinical branch."""
    def __init__(self, state):
        if state != dict(features=[], conditioning='mri_only'):
            raise ValueError('MRI-only NACC cannot import a clinical transform')
        self.state = state

    @classmethod
    def fit(cls, train):
        return cls(dict(features=[], conditioning='mri_only'))

    def __call__(self, row):
        return torch.empty(0, dtype=torch.float32)


class TrainingDataset(PairedT1FDGDataset):
    def __init__(self, rows, cfg, clinical):
        self.df = rows.reset_index(drop=True).copy()
        if not self.df.has_fdg_pet.astype(str).str.lower().eq('true').all():
            raise ValueError('Paired MRI/PET required')
        root = Path(cfg['dataset_root']).resolve()
        for col in ('t1_path', 'fdg_pet_path'):
            if not all(Path(p).resolve().is_relative_to(root) for p in self.df[col]):
                raise ValueError('Image outside configured source cohort root')
        self.spatial_size = tuple(cfg['spatial_size'])
        self.split = 'volumes'
        self.normalize_inputs = self.normalize_targets = False
        self.cache_dir = Path(cfg['cache_dir']) if cfg.get('cache_dir') else None
        self.clinical = clinical
        self.verify_geometry = cfg.get('verify_geometry', True)

    def _load_or_build_sample(self, row):
        if self.verify_geometry:
            mri, pet = nib.load(row.t1_path), nib.load(row.fdg_pet_path)
            if mri.shape[:3] != pet.shape[:3] or not np.allclose(mri.affine, pet.affine, rtol=0, atol=1e-4):
                raise ValueError(f'MRI/PET grids differ: {row.sample_id}')
        return super()._load_or_build_sample(row)

    def _cache_key(self, row):
        signature = []
        for key in ('t1_path', 'fdg_pet_path'):
            p = Path(row[key])
            stat = p.stat()
            signature.append([str(p.resolve()), stat.st_size, stat.st_mtime_ns])
        return hashlib.sha256(json.dumps([signature, self.spatial_size, 'raw-v1']).encode()).hexdigest()[:20]+'.pt'

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if not all(torch.isfinite(sample[k]).all() for k in ('input', 'target')):
            raise ValueError(f'Non-finite image: {sample["sample_id"]}')
        sample['brain_mask'] = (sample['input'].abs() > 1e-6).float()
        if not sample['brain_mask'].any():
            raise ValueError('Empty MRI foreground')
        sample['clinical'] = self.clinical(None)
        return sample
