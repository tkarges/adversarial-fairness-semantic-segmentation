from attacks.segpgd_mask2former import SegPGDMask2Former
from datasets.metadata import get_dataset_info
from datasets.loading import build_transformer_dataloaders

if __name__ == "__main__":
    num_classes, ignore_index, data_root, _, _ = get_dataset_info("cityscapes")
    train_loader, val_loader, _, _, _ = build_transformer_dataloaders(
        dataset="cityscapes",
        model="mask2former",
        data_root=data_root,
        batch_size=2,
        eval_batch_size=2,
        ignore_index=ignore_index,
        num_classes=num_classes,
        distributed=False
    )
    print(next(iter(train_loader)))
    attacker = SegPGDMask2Former(iterations=3, epsilon=8.0/255, alpha=2.0/255, num_classes=num_classes, ignore_index=ignore_index)
    
    