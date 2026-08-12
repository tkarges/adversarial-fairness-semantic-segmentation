import random
import numpy as np
from cityscapes import CityscapesDataset
from torch.utils.data import Subset
from utils.ddp_utils import dist_print, is_distributed, is_main_process
import torch
import os
from tqdm.auto import tqdm
from voc2012 import VOC2012Dataset
from pathlib import Path


def compute_image_histograms(dataset, num_classes, ignore_index, dataset_name):
    '''
    This function computes a histogram of shape [num_classes] for each sample in the dataset 
    '''
    hists = []
    
    for i in tqdm(range(len(dataset))):
        mask = dataset.get_raw_mask(i).cpu().numpy()
        # Class label 255 (ignore_index for Cityscapes) can be ignored
        valid = mask != ignore_index
        hist = np.bincount(mask[valid].reshape(-1), minlength=num_classes)
        hists.append(hist)
    
    # Contains one class histogram for each image in the dataset
    save_path = f'{Path(__file__).resolve().parents[1]}/datasets/histograms/training_hists_{dataset_name}.npy'
    np.save(save_path, np.stack(hists, axis=0)) 
    return np.stack(hists, axis=0)

def choose_distribution_matched_subset(
    dataset_name,
    weight_frac=0.05,
    num_trials=2000,
    seed=42,
):
    '''
    This samples a subset of size weight_frac * len(hists) from the histograms, such that the resulting
    subset closely resembles the label distribution in the overall set of histograms (the training set)     
    '''
    if is_main_process():
        dist_print('Creating subset for weight estimation')
        
    # Reproducibility
    rng = random.Random(seed)

    try:
        hists_path = f'{Path(__file__).resolve().parents[1]}/datasets/histograms/training_hists_{dataset_name}.npy'
        hists = np.load(hists_path)
    except Exception as e:
        print('Pre-compute image histograms or specify a valid path to the saved numpy array')

    n = hists.shape[0]
    m = max(1, int(round(n * weight_frac)))
    all_idx = list(range(n))

    # Summing up all image histograms and dividing by the sum of pixels yield the class distribution
    full_hist = hists.sum(axis=0).astype(np.float64)
    full_dist = full_hist / np.clip(full_hist.sum(), 1.0, None)

    best_weight_idx = None
    best_score = float("inf")

    for _ in tqdm(range(num_trials)):
        # Sample a subset from the entire dataset and calculate its class distribution
        sample_idx = rng.sample(all_idx, m)
        sample_hist = hists[sample_idx].sum(axis=0).astype(np.float64)
        sample_dist = sample_hist / np.clip(sample_hist.sum(), 1.0, None)

        # Computes how close the two distributions are
        score = np.abs(sample_dist - full_dist).sum()  # L1 distance

        # If the current sample is better than the best one so far update the best subset
        if score < best_score:
            best_score = score
            best_weight_idx = sample_idx

    weight_idx = sorted(best_weight_idx)
    main_idx = sorted(set(all_idx))
    
    save_split_indices(
        f'{Path(__file__).resolve().parents[1]}/datasets/splits/{dataset_name}_weight_split_idx.pt',
        train_main_idx=main_idx,
        train_weight_idx=weight_idx,
        weight_frac=weight_frac,
        num_trials=num_trials,
        seed=seed
    )
    
    summarize_subset_match(hists, weight_idx)
    
    return main_idx, weight_idx, best_score

def summarize_subset_match(hists, meta_idx):
    '''
    This can be used to check how good the subset is compared to the overall label distribution
    '''
    full_hist = hists.sum(axis=0).astype(np.float64)
    meta_hist = hists[meta_idx].sum(axis=0).astype(np.float64)

    full_dist = full_hist / full_hist.sum()
    meta_dist = meta_hist / meta_hist.sum()

    print("Full dist :", np.round(full_dist, 6))
    print("Meta dist :", np.round(meta_dist, 6))
    print("Abs diff  :", np.round(np.abs(full_dist - meta_dist), 6))
    print("L1 diff   :", np.abs(full_dist - meta_dist).sum())

def save_split_indices(path, train_main_idx, train_weight_idx, weight_frac, num_trials, seed):
    
    if is_main_process():
        dist_print('Saving split indices to disk')
        
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "train_main_idx": list(train_main_idx),
        "train_weight_idx": list(train_weight_idx),
        "weight_frac": weight_frac,
        "num_trials": num_trials,
        "seed": seed
    }
    torch.save(payload, path)

def load_split_indices(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload

def create_split(dataset, root_dir, crop_size, split_path=None):
    if dataset == 'cityscapes':
        dataset_raw = CityscapesDataset(root=root_dir, split="train", augment=False)
        dataset_aug = CityscapesDataset(root=root_dir, split="train", crop_size=crop_size, augment=True)
    elif dataset == 'voc2012':
        dataset_raw = VOC2012Dataset(root=root_dir, split='train', augment=False)
        dataset_aug = VOC2012Dataset(root=root_dir, split='train', augment=True)
    else:
        raise ValueError(f'Invalid dataset: {dataset}')

    if split_path is None:
        raise ValueError("create_split requires split_path in distributed mode")

    if is_main_process():
        dist_print("Creating weighting split of train")

        if split_path is not None and os.path.exists(split_path):
            dist_print("Using saved indices")
        else:
            train_main_idx, train_weight_idx, _ = choose_distribution_matched_subset(
                dataset_name=dataset,
                weight_frac=0.15,
                num_trials=1000000,
                seed=42
            )
            save_split_indices(
                split_path,
                train_main_idx=train_main_idx,
                train_weight_idx=train_weight_idx,
                weight_frac=0.15,
                num_trials=1000000,
                seed=42
            )

    if is_distributed():
        torch.distributed.barrier()

    saved_split = load_split_indices(split_path)
    train_main_idx = saved_split["train_main_idx"]
    train_weight_idx = saved_split["train_weight_idx"]

    #assert len(set(train_main_idx).intersection(set(train_weight_idx))) == 0
    #assert len(train_main_idx) + len(train_weight_idx) == len(dataset_raw)

    train_dataset = Subset(dataset_aug, train_main_idx)
    weight_dataset = Subset(dataset_raw, train_weight_idx)

    return train_dataset, weight_dataset

'''
if __name__ == '__main__':
    train_ds, weight_ds = create_split(
        dataset='voc2012',
        root_dir='/work/tkarges/voc2012-root',
        split_path='/home/tkarges/thesis-project/datasets/splits/voc2012_weight_split_idx.pt'
    )
    print(len(train_ds), len(weight_ds))
'''