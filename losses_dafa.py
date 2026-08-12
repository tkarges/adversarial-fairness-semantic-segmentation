import torch
import torch.nn.functional as F
    
def dafa_trades_loss(
    model,
    X,
    y,
    attacker,
    num_classes,
    ignore_index,
    class_weights=None,
    beta=1.0,
    aux_weight=0.4,
):
    '''
    DAFA-TRADES loss from the DAFA paper
    ''' 
    out_clean = model(X, return_aux=True)

    if isinstance(out_clean, tuple):
        logits_clean, aux_logits = out_clean
    else:
        logits_clean, aux_logits = out_clean, None

    X_adv = attacker.attack(
        model=model,
        X=X,
        mask=y,
        clean_logits=logits_clean.detach(),
        class_weights=class_weights,
    )

    out_adv = model(X_adv, return_aux=False)
    logits_adv = out_adv[0] if isinstance(out_adv, tuple) else out_adv

    valid = ((y != ignore_index) & (y >= 0) & (y < num_classes))

    # Clean cross-entropy
    clean_ce_map = F.cross_entropy(
        logits_clean,
        y,
        ignore_index=ignore_index,
        reduction="none",
    )

    if class_weights is None:
        # Weights not active
        loss_clean = clean_ce_map[valid].mean()
    else:
        # Weights active
        pixel_weights = class_weights[y[valid]].to(clean_ce_map.dtype)
        loss_clean = (clean_ce_map[valid] * pixel_weights).mean()

    clean_probs = F.softmax(logits_clean.detach(), dim=1)

    # TRADES KL-divergence loss between clean and adversarial output distributions
    kl_map = F.kl_div(
        F.log_softmax(logits_adv, dim=1), 
        clean_probs,
        reduction="none",
    ).sum(dim=1)

    loss_robust = kl_map[valid].mean()

    # Models have an auxiliary output
    loss_aux = logits_clean.new_tensor(0.0)

    if aux_logits is not None:
        loss_aux = F.cross_entropy(
            aux_logits,
            y,
            ignore_index=ignore_index,
            weight=class_weights,
        )
    
    loss = loss_clean + beta * loss_robust + aux_weight * loss_aux

    return loss

def dafa_dense_loss(
    model,
    X_clean,
    X_adv_base,
    y_clean,
    y_adv,
    attacker,
    num_classes,
    ignore_index,
    class_weights=None,
    adv_weight=1.0,
    aux_weight=0.4
):
    '''
    SegPGD-specific DAFA-CE formulation loss
    '''
    out_clean = model(X_clean, return_aux=True)
    
    if isinstance(out_clean, tuple):
        logits_clean, aux_logits = out_clean
    else:
        logits_clean, aux_logits = out_clean, None
    
    X_adv = attacker.attack(
        model=model,
        X=X_adv_base,
        mask=y_adv,
        class_weights=class_weights,
    )
    
    out_adv = model(X_adv, return_aux=False)
    logits_adv = out_adv[0] if isinstance(out_adv, tuple) else out_adv
    
    valid = ((y_clean != ignore_index) & (y_clean >= 0) & (y_clean < num_classes))
    
    # Clean cross-entropy
    clean_ce_map = F.cross_entropy(
        logits_clean,
        y_clean,
        ignore_index=ignore_index,
        reduction="none",
    )
    
    # Weighted or unweighted, depending on training state
    if class_weights is None:
        loss_clean = clean_ce_map[valid].mean()
    else:
        pixel_weights = class_weights[y_clean[valid]].to(clean_ce_map.dtype)
        loss_clean = (clean_ce_map[valid] * pixel_weights).mean()
    
    # Main difference: adv loss is now also cross-entropy
    adv_ce_map = F.cross_entropy(
        logits_adv,
        y_clean,
        ignore_index=ignore_index,
        reduction="none"
    )
    
    loss_robust = adv_ce_map[valid].mean()
    
    loss_aux = logits_clean.new_tensor(0.0)
    
    if aux_logits is not None:
        loss_aux = F.cross_entropy(
            aux_logits,
            y_clean,
            ignore_index=ignore_index,
            weight=class_weights,
        )
        
    loss = loss_clean + adv_weight * loss_robust + aux_weight * loss_aux
    
    return loss
    
@torch.no_grad()
def update_dafa_stats(
    logits,
    target,
    prob_sum,
    class_counts,
    num_classes,
    ignore_index
):
    '''
    Collects class-wise probabilities and class occurence for one batch and adds
    it to the corresponding global variables
    '''
    probs = logits.softmax(dim=1)
    probs = probs.permute(0, 2, 3, 1).reshape(-1, num_classes)
    labels = target.reshape(-1)

    valid = ((labels != ignore_index) & (labels >= 0) & (labels < num_classes))

    labels = labels[valid]
    probs = probs[valid]

    prob_sum.index_add_(0, labels, probs.double())
    class_counts += torch.bincount(labels, minlength=num_classes).double()

@torch.no_grad()
def collect_dafa_statistics(
    model,
    dataloader,
    attacker,
    prob_sum,
    class_counts,
    num_classes,
    ignore_index,
    device
):
    '''
    Collects statistics needed for DAFA weight computation
    '''
    model.eval()

    prob_sum.zero_()
    class_counts.zero_()

    for X, y in dataloader:
        X = X.to(device, non_blocking=device.type == "cuda")
        y = y.to(device, non_blocking=device.type == "cuda")

        with torch.no_grad():
            out_clean = model(X, return_aux=False)
            clean_logits = out_clean[0] if isinstance(out_clean, tuple) else out_clean

        with torch.enable_grad():
            X_adv = attacker.attack(
                model=model,
                X=X,
                mask=y,
                clean_logits=clean_logits.detach(),
                class_weights=None,
            )

        with torch.no_grad():
            out_adv = model(X_adv, return_aux=False)
            logits_adv = out_adv[0] if isinstance(out_adv, tuple) else out_adv

            update_dafa_stats(
                logits=logits_adv,
                target=y,
                prob_sum=prob_sum,
                class_counts=class_counts,
                num_classes=num_classes,
                ignore_index=ignore_index,
            )

def compute_dafa_weights(
    prob_sum,
    class_counts,
    dafa_lambda=0.1,
    min_class_count=1,
    min_weight=0.25,
    max_weight=2.0
):
    '''
    Computes class weights for DAFA
    '''
    valid_classes = class_counts >= min_class_count

    p = torch.zeros_like(prob_sum, dtype=torch.float64)

    # DAFA similarity matrix
    p[valid_classes] = prob_sum[valid_classes] / class_counts[valid_classes].unsqueeze(1)

    p = p.float()
    diag = p.diag()

    harder = diag[:, None] < diag[None, :]
    easier = diag[:, None] > diag[None, :]

    valid_pairs = valid_classes[:, None] & valid_classes[None, :]

    positive = (harder & valid_pairs).to(p.dtype) * p * diag[None, :]
    negative = (easier & valid_pairs).to(p.dtype) * p.T * diag[:, None]

    weights = 1.0 + dafa_lambda * (positive.sum(dim=1) - negative.sum(dim=1))
    weights = torch.where(valid_classes, weights, torch.ones_like(weights))
    weights = weights.clamp(min=min_weight, max=max_weight)

    return weights