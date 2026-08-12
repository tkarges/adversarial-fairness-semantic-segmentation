from cityscapes import CityscapesDataset
from voc2012 import VOC2012Dataset
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import torch

def build_dataset(dataset, root, split, prob_class_crop, use_rl=False, split_idx_path=None):
    if not use_rl:
        augment = (split == "train")
        if dataset == "cityscapes":
            return CityscapesDataset(root=root, split=split, augment=augment, prob_class_crop=prob_class_crop)
        elif dataset == "voc2012":
            return VOC2012Dataset(root=root, split=split, augment=augment, prob_class_crop=prob_class_crop)
        else:
            ValueError(f"Unsupported dataset: {dataset}")
    else:
        if split == "train":
            # RL needs a separate policy set; the indices for this set are read from disk to save time
            split_idx = torch.load(split_idx_path, map_location="cpu", weights_only=True)
            train_idx = split_idx["train_main_idx"]
            policy_idx = split_idx["train_weight_idx"]
            dataset_raw = CityscapesDataset(root=root, split="train", augment=False, prob_class_crop=0.0)
            dataset_augmented = CityscapesDataset(root=root, split="train", augment=True, prob_class_crop=prob_class_crop)
            policy_dataset = Subset(dataset=dataset_raw, indices=policy_idx)
            train_dataset = Subset(dataset=dataset_augmented, indices=train_idx)
            return train_dataset, policy_dataset
        else:
            return CityscapesDataset(root=root, split="val", augment=True, prob_class_crop=0.0)
            
        

def build_dense_dataloaders(
    dataset, 
    data_root, 
    distributed,
    batch_size=8,
    eval_batch_size=2,
    prob_class_crop=0.0,
    use_rl=False,
    split_idx_path=None
):
    policy_dataset = None
    if use_rl:
        train_dataset, policy_dataset = build_dataset(
            dataset=dataset,
            root=data_root,
            split="train",
            prob_class_crop=prob_class_crop,
            use_rl=True,
            split_idx_path=split_idx_path
        )
    else:
        train_dataset = build_dataset(
            dataset=dataset, 
            root=data_root, 
            split="train", 
            prob_class_crop=prob_class_crop,
            use_rl=False,
            split_idx_path=None
        )
        
    val_dataset = build_dataset(
        dataset=dataset, 
        root=data_root, 
        split="val", 
        prob_class_crop=0.0,
        use_rl=False,
        split_idx_path=None
    )
    
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler = DistributedSampler(val_dataset) if distributed else None
    policy_sampler = DistributedSampler(policy_dataset) if distributed and policy_dataset is not None else None
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=eval_batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True
    )
    
    policy_loader = None
    if policy_dataset is not None:
        policy_loader = DataLoader(
            dataset=policy_dataset,
            batch_size=eval_batch_size,
            sampler=policy_sampler,
            shuffle=False,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True
        )
    
    return (
        train_loader,
        val_loader,
        train_sampler,
        val_sampler,
        policy_loader,
        policy_sampler,
    )