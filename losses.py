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
    '''
    Pixel-wise cross-entropy loss for semantic segmentation
    '''
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
            align_corners=False
        )

    valid_mask = ((y != ignore_index) & (y >= 0) & (y < num_classes))

    loss_map = F.cross_entropy(
        logits,
        y,
        weight=None,
        ignore_index=ignore_index,
        reduction="none"
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
                align_corners=False
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
    '''
    Clean training requires only a clean cross-entropy loss
    '''
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
    '''
    Standard adversarial training uses a linear combination of clean and adversarial loss 
    '''
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
    '''
    Computes the losses for models trained with only adversarial examples
    '''
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