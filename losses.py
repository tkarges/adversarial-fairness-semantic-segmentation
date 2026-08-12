import torch
import torch.nn.functional as F
from utils.ddp_utils import dist_print

def ce_loss(
    model,
    X,
    y,
    num_classes,
    ignore_index,
    aux_weight=0.4,
    return_logits=False
):
    out = model(X, return_aux=True)

    if isinstance(out, tuple):
        logits, aux_logits = out
    else:
        logits, aux_logits = out, None

    if logits.shape[-2:] != y.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=y.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    valid_mask = ((y != ignore_index) & (y >= 0) & (y < num_classes))

    loss_map = F.cross_entropy(
        logits,
        y,
        weight=None,
        ignore_index=ignore_index,
        reduction="none",
    )

    if valid_mask.any():
        main_ce = loss_map[valid_mask].mean()
    else:
        main_ce = logits.sum() * 0.0

    aux_ce = logits.new_tensor(0.0)

    if aux_logits is not None:
        if aux_logits.shape[-2:] != y.shape[-2:]:
            aux_logits = F.interpolate(
                aux_logits,
                size=y.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        aux_ce = F.cross_entropy(
            aux_logits,
            y,
            ignore_index=ignore_index
        )

    loss = main_ce + aux_weight * aux_ce
    
    return loss

def clean_training_loss(
    model,
    X,
    y,
    num_classes,
    ignore_index
):
    amp_enabled = X.is_cuda
    device_type = X.device.type
        
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp_enabled):
        loss = ce_loss(
            model=model,
            X=X,
            y=y,
            num_classes=num_classes,
            ignore_index=ignore_index
        )
                
    return loss
    

def adversarial_training_loss(
    model, 
    X, 
    X_adv, 
    y, 
    num_classes, 
    ignore_index,
    adversarial_weight=1.0,
): 
    loss_clean = ce_loss(
        model=model,
        X=X,
        y=y,
        num_classes=num_classes,
        ignore_index=ignore_index
    )
      
    loss_adv = ce_loss(
        model=model,
        X=X_adv,
        y=y,
        num_classes=num_classes,
        ignore_index=ignore_index
    )
    
    # Final loss is a linear combination of clean loss and adversarial loss
    loss = loss_clean + adversarial_weight * loss_adv
    
    return loss

def adversarial_training_50_50_loss(model, X_clean, X_adv, y_clean, y_adv, num_classes, ignore_index):
    """
    Computes the 50/50 adversarial training objective.

    The loss combines the clean segmentation loss and the adversarial
    segmentation loss with equal weight. This objective is used for the
    at_50_50 experiment setting in the thesis.

    Args:
        model: Segmentation model returning logits of shape [B, C, H, W].
        X_clean: Clean input batch.
        X_adv: Adversarially perturbed input batch.
        y: Ground-truth segmentation mask.
        ignore_index: Label value excluded from the loss.

    Returns:
        Scalar training loss.
    """
    if X_clean.device != X_adv.device:
        raise ValueError(f'Clean and adversarial samples need to be on the same device, \
            got X_clean: {X_clean.device} and X_adv: {X_adv.device}')
        
    amp_enabled = X_adv.is_cuda and X_clean.is_cuda
    device_type = X_clean.device.type
    
    X_combined = torch.cat([X_clean, X_adv], dim=0)
    y_combined = torch.cat([y_clean, y_adv], dim=0).long()
    
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp_enabled):
        loss = ce_loss(
            model=model,
            X=X_combined,
            y=y_combined,
            num_classes=num_classes,
            ignore_index=ignore_index
        )
        #print(loss)
        
    return loss


def adversarial_training_100_loss(model, X_adv, y, num_classes, ignore_index):
    amp_enabled = X_adv.is_cuda
    device_type = X_adv.device.type
    
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp_enabled):
        loss = ce_loss(
            model=model,
            X=X_adv,
            y=y,
            num_classes=num_classes,
            ignore_index=ignore_index
        )
            
    return loss


'''
    The following loss functions are used for training Mask2Former.
    Since transformer models work quite differently than dense model, especially regarding
    the expected format of input data, these are defined separately.
'''

def cast_auxiliary_predictions_to_float(auxiliary_predictions):
    if auxiliary_predictions is None:
        return None
    
    casted = []
    
    for pred in auxiliary_predictions:
        casted.append({key: value.float() if torch.is_tensor(value) else value for key, value in pred.items()})
        
    return casted

def mask2former_native_loss(
    model, 
    criterion, 
    pixel_values,
    pixel_mask, 
    mask_labels, 
    class_labels
):
    outputs = model(
        pixel_values=pixel_values,
        pixel_mask=pixel_mask
    )
    
    loss_dict = criterion(
        masks_queries_logits=outputs.masks_queries_logits.float(),
        class_queries_logits=outputs.class_queries_logits.float(),
        mask_labels=mask_labels,
        class_labels=class_labels,
        auxiliary_predictions=cast_auxiliary_predictions_to_float(outputs.auxiliary_logits)
    )
    
    loss = sum(
        criterion.weight_dict.get(name, 1.0) * value 
        for name, value in loss_dict.items()
    )
    
    return loss, loss_dict, outputs

def clean_training_loss_transformer(
    model,
    criterion,
    pixel_values,
    pixel_mask,
    mask_labels,
    class_labels
):
    loss, _, _ = mask2former_native_loss(
        model=model,
        criterion=criterion,
        pixel_values=pixel_values,
        pixel_mask=pixel_mask,
        mask_labels=mask_labels,
        class_labels=class_labels
    )
    
    return loss

def adversarial_training_loss_transformer(
    model,
    attacker,
    pixel_values,
    pixel_mask,
    mask_labels,
    class_labels,
    segmentation_map,
    criterion_clean,
    criterion_adv,
    adv_weight
):
    pixel_values_adv = attacker.attack(
        model=model,
        pixel_values=pixel_values,
        segmentation_map=segmentation_map,
        pixel_mask=pixel_mask
    )
    
    clean_loss, _, _ = mask2former_native_loss(
        model=model,
        criterion=criterion_clean,
        pixel_values=pixel_values,
        pixel_mask=pixel_mask,
        mask_labels=mask_labels,
        class_labels=class_labels
    )
    
    adv_loss, _, _ = mask2former_native_loss(
        model=model,
        criterion=criterion_adv,
        pixel_values=pixel_values_adv,
        pixel_mask=pixel_mask,
        mask_labels=mask_labels,
        class_labels=class_labels
    )
    
    return clean_loss + adv_weight * adv_loss

def split_transformer_batch(
    pixel_values,
    pixel_mask,
    mask_labels,
    class_labels,
    segmentation_map
):
    batch_size = pixel_values.size(0)
    adv_size = batch_size // 2
        
    perm = torch.randperm(batch_size, device=pixel_values.device)
    adv_idx = perm[:adv_size]
    clean_idx = perm[adv_size:]
        
    clean_pixel_values = pixel_values[clean_idx]
    adv_pixel_values = pixel_values[adv_idx]
        
    clean_segmentation_map = segmentation_map[clean_idx]
    adv_segmentation_map = segmentation_map[adv_idx]
        
    if pixel_mask is not None:
        clean_pixel_mask = pixel_mask[clean_idx]
        adv_pixel_mask = pixel_mask[adv_idx]
    else:
        clean_pixel_mask = None
        adv_pixel_mask = None
            
    clean_idx_list = clean_idx.detach().cpu().tolist()
    adv_idx_list = adv_idx.detach().cpu().tolist()
        
    clean_mask_labels = [mask_labels[i] for i in clean_idx_list]
    clean_class_labels = [class_labels[i] for i in clean_idx_list]
        
    adv_mask_labels = [mask_labels[i] for i in adv_idx_list]
    adv_class_labels = [class_labels[i] for i in adv_idx_list]
        
    clean_batch = {
        "pixel_values": clean_pixel_values,
        "pixel_mask": clean_pixel_mask,
        "mask_labels": clean_mask_labels,
        "class_labels": clean_class_labels,
        "segmentation_map": clean_segmentation_map
    }
        
    adv_batch = {
        "pixel_values": adv_pixel_values,
        "pixel_mask": adv_pixel_mask,
        "mask_labels": adv_mask_labels,
        "class_labels": adv_class_labels,
        "segmentation_map": adv_segmentation_map
    }
        
    return clean_batch, adv_batch
    
def adversarial_training_50_50_loss_transformer(
    model,
    attacker,
    pixel_values,
    pixel_mask,
    mask_labels,
    class_labels,
    segmentation_map,
    criterion
):
    clean_batch, adv_batch = split_transformer_batch(
        pixel_values=pixel_values,
        pixel_mask=pixel_mask,
        mask_labels=mask_labels,
        class_labels=class_labels,
        segmentation_map=segmentation_map
    )
    
    pixel_values_adv = attacker.attack(
        model=model,
        pixel_values=adv_batch["pixel_values"],
        segmentation_map=adv_batch["segmentation_map"],
        pixel_mask=adv_batch["pixel_mask"]
    )
    
    pixel_values_combined = torch.cat([clean_batch["pixel_values"], pixel_values_adv], dim=0)
    
    if clean_batch["pixel_mask"] is None and adv_batch["pixel_mask"] is None:
        pixel_mask_combined = None
    else:
        pixel_mask_combined = torch.cat([clean_batch["pixel_mask"], adv_batch["pixel_mask"]], dim=0)
            
    mask_labels_combined = clean_batch["mask_labels"] + adv_batch["mask_labels"]
    class_labels_combined = clean_batch["class_labels"] + adv_batch["class_labels"]
        
    loss, _, _ = mask2former_native_loss(
        model=model,
        criterion=criterion,
        pixel_values=pixel_values_combined,
        pixel_mask=pixel_mask_combined,
        mask_labels=mask_labels_combined,
        class_labels=class_labels_combined
    )
        
    return loss

def adversarial_training_100_loss_transformer(
    model,
    attacker,
    pixel_values,
    pixel_mask,
    mask_labels,
    class_labels,
    segmentation_map,
    criterion
):
    pixel_values_adv = attacker.attack(
        model=model,
        pixel_values=pixel_values,
        segmentation_map=segmentation_map,
        pixel_mask=pixel_mask
    )
        
    loss, _, _ = mask2former_native_loss(
        model=model,
        criterion=criterion,
        pixel_values=pixel_values_adv,
        pixel_mask=pixel_mask,
        mask_labels=mask_labels,
        class_labels=class_labels
    )
        
    return loss