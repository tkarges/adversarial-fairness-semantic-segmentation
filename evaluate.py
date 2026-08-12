import torch
import torch.distributed as dist
import torch.nn.functional as F
import time
from utils.ddp_utils import is_distributed, is_main_process, dist_print, reduce_sum
from tqdm.auto import tqdm

@torch.no_grad()
def model_prediction_bias(logits, targets, num_classes, ignore_index=255, class_subset=None):
    if targets.ndim == 2:
        targets = targets.unsqueeze(0)

    if targets.ndim == 4:
        targets = targets.squeeze(1)

    targets = targets.to(logits.device)

    probs = logits.softmax(dim=1)

    valid = (
        (targets >= 0)
        & (targets < num_classes)
        & (targets != ignore_index)
    )

    if class_subset is not None:
        class_subset = torch.as_tensor(
            class_subset,
            device=targets.device,
            dtype=targets.dtype,
        )
        mask = valid & torch.isin(targets, class_subset)
    else:
        mask = valid

    probs_flat = probs.permute(0, 2, 3, 1)[mask]
    targets_flat = targets[mask].long()

    targets_onehot = F.one_hot(
        targets_flat,
        num_classes=num_classes,
    ).float()

    gn = (probs_flat * (1.0 - targets_onehot)).sum(dim=0)
    gp = (probs_flat * targets_onehot).sum(dim=0)
    cf = targets_onehot.sum(dim=0)

    return gn, gp, cf

def build_confusion_matrix(
    pred,
    target,
    num_classes,
    ignore_index
):
    pred = pred.view(-1).to(torch.int64)
    target = target.view(-1).to(torch.int64)

    if ignore_index is not None:
        mask = target != ignore_index
        pred = pred[mask]
        target = target[mask]

    mask = (target >= 0) & (target < num_classes)
    pred = pred[mask]
    target = target[mask]

    idx = target * num_classes + pred
    conf = torch.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)
    return conf

def compute_accuracy(confusion_matrix, eps=1e-7):
    tp = torch.diag(confusion_matrix).float()
    fn = confusion_matrix.sum(dim=1).float() - tp
    
    acc_per_class = tp / (tp + fn + eps)
    valid = (tp + fn) > 0
    macc = acc_per_class[valid].mean()
    
    return macc, acc_per_class, valid

def compute_precision(confusion_matrix, eps=1e-7):
    tp = torch.diag(confusion_matrix).float()
    fp = confusion_matrix.sum(dim=0).float() - tp
    
    acc_per_class = tp / (tp + fp + eps)
    valid = (tp + fp) > 0
    macc = acc_per_class[valid].mean()
    
    return macc, acc_per_class, valid

def compute_iou(confusion_matrix, eps=1e-7): 
    tp = torch.diag(confusion_matrix).float()
    fp = confusion_matrix.sum(dim=0).float() - tp
    fn = confusion_matrix.sum(dim=1).float() - tp
    
    union = tp + fp + fn
    
    iou_per_class = tp / (union + eps)
    valid = union > 0
    miou = iou_per_class[valid].mean()
    
    return miou, iou_per_class, valid

def compute_iou_normalized(confusion_matrix, num_classes, eps=1e-7): 
    tp = torch.diag(confusion_matrix).float()
    fp = confusion_matrix.sum(dim=0).float() - tp
    fn = confusion_matrix.sum(dim=1).float() - tp
    
    fp_norm = fp / (num_classes - 1)
    fn_norm = fn / (num_classes - 1)
    
    union = tp + fp_norm + fn_norm
    
    iou_per_class = tp / (union + eps)
    valid = union > 0
    miou = iou_per_class[valid].mean()
    
    return miou, iou_per_class, valid

def evaluate(
    model, 
    dataloader,
    num_classes,
    ignore_index,
    metric='iou', 
    attacker=None,
    device=None,
    eval_logger=None,
    compute_pred_bias=False,
    easy_classes=None,
    hard_classes=None,
    model_name=None
):
    if is_main_process():
        attacker_name = attacker.__class__.__name__ if attacker is not None else "None"
        dist_print('\nEntered Evaluation')
        dist_print(f'\tModel: {model.__class__.__name__} Attacker: {attacker_name}, Metric: {metric}')
        eval_start_time = time.time()
    
    if device is None:
        device = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')
    
    model.eval()
    
    if compute_pred_bias:
        gn_global = torch.zeros(num_classes, device=device)
        gp_global = torch.zeros(num_classes, device=device)
        cf_global = torch.zeros(num_classes, device=device)
        gn_easy = torch.zeros(num_classes, device=device)
        gp_easy = torch.zeros(num_classes, device=device)
        cf_easy = torch.zeros(num_classes, device=device)
        gn_hard = torch.zeros(num_classes, device=device)
        gp_hard = torch.zeros(num_classes, device=device)
        cf_hard = torch.zeros(num_classes, device=device)
    else:
        gn_global = None
        gp_global = None
        cf_global = None
        gn_easy = None
        gp_easy = None
        cf_easy = None
        gn_hard = None
        gp_hard = None
        cf_hard = None
        
    conf_global = torch.zeros((num_classes, num_classes), device=device, dtype=torch.int64)

    iterator = tqdm(dataloader, desc='Evaluating', disable=not is_main_process(), leave=False)
    
    val_loss = 0.0
    num_samples = 0
    
    for X, y in iterator:
        
        X = X.to(device)
        y = y.to(device)
        
        if attacker is not None:
            with torch.enable_grad():
                X_input = attacker.attack(model, X, y)
        else:
            X_input = X
        
        '''
        amp_enabled = X_input.is_cuda
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp_enabled):
                logits = model(X_input)
                pred = logits.argmax(dim=1)
        '''
        with torch.no_grad():
            logits = model(X_input)
            pred = logits.argmax(dim=1)
        
        loss = F.cross_entropy(logits, y, ignore_index=255)
        val_loss += loss.item() * X.size(0)
        num_samples += X.size(0)
        
        conf_global += build_confusion_matrix(pred=pred, target=y, num_classes=num_classes, ignore_index=ignore_index)
        
        if compute_pred_bias:
            gn, gp, cf = model_prediction_bias(logits, y, num_classes=num_classes, ignore_index=ignore_index)
            gn_global += gn
            gp_global += gp
            cf_global += cf
            
            if easy_classes is not None:
                gn, gp, cf = model_prediction_bias(
                    logits=logits, 
                    targets=y, 
                    num_classes=num_classes, 
                    ignore_index=ignore_index, 
                    class_subset=easy_classes
                )
                gn_easy += gn
                gp_easy += gp
                cf_easy += cf
            
            if hard_classes is not None:
                gn, gp, cf = model_prediction_bias(
                    logits=logits, 
                    targets=y, 
                    num_classes=num_classes, 
                    ignore_index=ignore_index, 
                    class_subset=hard_classes
                )
                gn_hard += gn
                gp_hard += gp
                cf_hard += cf
              
    if is_distributed():
        dist.all_reduce(conf_global, op=dist.ReduceOp.SUM)
        if compute_pred_bias:
            dist.all_reduce(gn_global, op=dist.ReduceOp.SUM)
            dist.all_reduce(gp_global, op=dist.ReduceOp.SUM)
            dist.all_reduce(cf_global, op=dist.ReduceOp.SUM)
            if easy_classes is not None:
                dist.all_reduce(gn_easy, op=dist.ReduceOp.SUM)
                dist.all_reduce(gp_easy, op=dist.ReduceOp.SUM)
                dist.all_reduce(cf_easy, op=dist.ReduceOp.SUM)
            if hard_classes is not None:
                dist.all_reduce(gn_hard, op=dist.ReduceOp.SUM)
                dist.all_reduce(gp_hard, op=dist.ReduceOp.SUM)
                dist.all_reduce(cf_hard, op=dist.ReduceOp.SUM)            
    
    if eval_logger is not None:
        tp = torch.diag(conf_global).float()
        fp = conf_global.sum(dim=0).float() - tp
        fn = conf_global.sum(dim=1).float() - tp
        eval_logger.log(tp, fp, fn)
    
    if metric == 'accuracy':
        mean_score, per_class_score, valid = compute_accuracy(conf_global)
    
    elif metric == 'iou':
        mean_score, per_class_score, valid = compute_iou(conf_global)
        
    elif metric == 'iou_normalized':
        mean_score, per_class_score, valid = compute_iou_normalized(conf_global)
        
    elif metric == 'precision':
        mean_score, per_class_score, valid = compute_precision(conf_global)
        
    else:
        return None
        
    if is_main_process():
        eval_time = time.time() - eval_start_time
        dist_print(f'Finished evaluation after {eval_time:.2f} seconds\n')
    
    if is_distributed():
        total_val_loss = reduce_sum(val_loss, device)
        total_num_samples = reduce_sum(num_samples, device)
        avg_loss = total_val_loss / max(total_num_samples, 1)
    else:
        avg_loss = val_loss / num_samples
    
    if is_main_process():
        if attacker is not None:
            dist_print(f'Robust validation loss: {avg_loss}')
        else:
            dist_print(f'Clean validation loss: {avg_loss}')
    
    return {
        'confusion_matrix': conf_global,
        'per_class_score': per_class_score,
        'mean_score': mean_score,
        'valid': valid,
        'gn': gn_global,
        'gp': gp_global,
        'cf': cf_global,
        'gn_easy': gn_easy,
        'gp_easy': gp_easy,
        'cf_easy': cf_easy,
        'gn_hard': gn_hard,
        'gp_hard': gp_hard,
        'cf_hard': cf_hard,
    }