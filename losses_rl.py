import torch
import torch.nn.functional as F

def weighted_pixel_ce(
    logits,
    y,
    num_classes,
    ignore_index,
    class_weights
):
    '''
    Computes class weighted cross-entropy loss.
    '''
    valid_mask = ((y != ignore_index) & (y < num_classes) & (y >= 0))
    
    # Fallback for invalid inputs
    if not valid_mask.any():
        return logits.float().sum() * 0.0
    
    # If class weights are used, they have to be on the correct device and have the correct datatype
    if class_weights is not None:
        class_weights = class_weights.to(
            device=logits.device,
            dtype=torch.float32
        )
        
        # Weights must not be infinite
        if not torch.isfinite(class_weights).all():
            raise ValueError(f"Infinite class weights detected: {class_weights}")
        
        # Weights must be positive
        if (class_weights <= 0).any():
            raise ValueError(f"All class weights have to be positive: {class_weights}")
    
    loss_map = F.cross_entropy(
        input=logits,
        target=y,
        weight=class_weights,
        ignore_index=ignore_index,
        reduction="none"
    )
    
    valid_losses = loss_map[valid_mask].float()
    
    if class_weights is None:
        return valid_losses.mean()
    
    pixel_weights = class_weights[y[valid_mask]].float()
    
    return valid_losses.sum() / pixel_weights.sum().clamp_min(1e-8)

def macro_averaged_class_ce(
    logits,
    y,
    num_classes,
    ignore_index, 
    class_weights=None
):
    '''
    Computes the macro-averaged loss for the RL framework
    '''
    valid_mask = ((y != ignore_index) & (y >= 0) & (y < num_classes))
    
    if not valid_mask.any():
        return logits.float().sum() * 0.0
    
    if class_weights is not None:
        class_weights = class_weights.to(
            device=logits.device,
            dtype=torch.float32,
        )

        if not torch.isfinite(class_weights).all():
            raise RuntimeError(
                f"Non-finite class weights: {class_weights}"
            )

        if (class_weights <= 0).any():
            raise RuntimeError(
                f"Non-positive class weights: {class_weights}"
            )
            
    loss_map = F.cross_entropy(
        input=logits,
        target=y,
        ignore_index=ignore_index,
        reduction="none"
    )
    
    class_losses = []
    class_level_weights = []
    
    for class_idx in range(num_classes):
        class_mask = valid_mask & (y == class_idx)
        
        if not class_mask.any():
            continue
        
        class_loss = loss_map[class_mask].mean()
        class_losses.append(class_loss)
        
        if class_weights is None:
            class_level_weights.append(class_loss.new_tensor(0.0))
        else:
            class_level_weights.append(class_weights[class_idx])
            
    if not class_losses:
        return logits.float().sum() * 0.0
    
    # Losses are now a list containing one element per class --> equal importance
    class_losses = torch.stack(class_losses)
    class_level_weights = torch.stack(class_level_weights).float()
    
    return (class_losses * class_level_weights).sum() / class_level_weights.sum().clamp_min(1e-8)

def ce_loss(
    model,
    X,
    y,
    num_classes,
    ignore_index,
    class_weights=None,
    aux_weight=0.4,
    return_logits=False,
    weight_mode="pixel"
):
    '''
    General loss function that can use either pixel or macro loss
    '''
    
    if weight_mode == "pixel":
        loss_fn = weighted_pixel_ce
    elif weight_mode == "macro":
        loss_fn = macro_averaged_class_ce
    else:
        raise ValueError(f"Invalid weighting scheme: {weight_mode}")
    
    out = model(X, return_aux=True)

    if isinstance(out, tuple):
        logits, aux_logits = out
    else:
        logits, aux_logits = out, None

    main_ce = loss_fn(
        logits=logits,
        y=y,
        num_classes=num_classes,
        ignore_index=ignore_index,
        class_weights=class_weights
    )

    aux_ce = logits.float().new_zeros(())

    if aux_logits is not None:
        aux_ce = loss_fn(
            logits=aux_logits,
            y=y,
            num_classes=num_classes,
            ignore_index=ignore_index,
            class_weights=class_weights
        )

    loss = main_ce + aux_weight * aux_ce

    if return_logits:
        return loss, logits

    return loss

def adversarial_training_loss(
    model,
    X_clean,
    X_adv,
    y_clean,
    y_adv,
    num_classes,
    ignore_index,
    class_weights=None,
    class_weights_adv=None,
    clean_weight_mode="pixel",
    adv_weight_mode="pixel",
    adversarial_weight=1.0
):
    '''
    Combined clean and adversarial loss as overall objective for the RL approaches
    '''
    amp_enabled = X_clean.is_cuda and X_adv.is_cuda
    device_type = X_clean.device.type
    
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp_enabled):
        loss_clean = ce_loss(
            model=model,
            X=X_clean,
            y=y_clean,
            num_classes=num_classes,
            ignore_index=ignore_index,
            class_weights=class_weights,
            weight_mode=clean_weight_mode
        )

        loss_adv = ce_loss(
            model=model,
            X=X_adv,
            y=y_adv,
            num_classes=num_classes,
            ignore_index=ignore_index,
            class_weights=class_weights_adv,
            weight_mode=adv_weight_mode
        )

    return 0.5 * loss_clean + 0.5 * adversarial_weight * loss_adv
