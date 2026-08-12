import argparse
import os
import time
from evaluate import evaluate
from utils.ddp_utils import is_main_process, dist_print, reduce_sum
from utils.checkpoint import save_checkpoint
from utils.logging import MetricsLogger, EvalLogger, CleanLogger
from losses import (
    clean_training_loss, 
    adversarial_training_loss, 
    adversarial_training_50_50_loss, 
    adversarial_training_100_loss,
)
from utils.stats import print_per_class_evolution

import torch
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
            
def split_clean_adv_batch(X, y):
    '''
    Splits batch for 50/50 SegPGD training
    '''
    batch_size = X.size(0)
    perm = torch.randperm(batch_size, device=X.device)

    adv_size = batch_size // 2
    adv_idx = perm[:adv_size]
    clean_idx = perm[adv_size:]

    X_adv_base = X[adv_idx]
    y_adv = y[adv_idx]

    X_clean = X[clean_idx]
    y_clean = y[clean_idx]

    return X_clean, y_clean, X_adv_base, y_adv

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    num_classes,
    ignore_index,
    attacker=None,
    adv_train=False,
    adv_mode=None,
    adv_weight=1.0
):
    epoch_loss = 0.0
    num_samples = 0
    
    for batch_idx, (X, y) in enumerate(dataloader):
        X = X.to(device, non_blocking=device.type == 'cuda')
        y = y.to(device, non_blocking=device.type == 'cuda')
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward passes are implemented in the loss functions
        if adv_mode is None:
            if adv_train:
                X_input = X
                X_adv = attacker.attack(model, X, y)
                loss = adversarial_training_loss(
                    model=model,
                    X=X_input,
                    X_adv=X_adv,
                    y=y,
                    ignore_index=ignore_index,
                    num_classes=num_classes,
                    adversarial_weight=adv_weight
                )
            else:
                loss = clean_training_loss(
                    model=model, 
                    X=X, 
                    y=y,
                    num_classes=num_classes,
                    ignore_index=ignore_index 
                )
                
        elif adv_mode == 'at_50_50':
            X_clean, y_clean, X_adv_base, y_adv = split_clean_adv_batch(X, y)
            X_adv = attacker.attack(model, X_adv_base, y_adv)
            loss = adversarial_training_50_50_loss(
                model=model, 
                X_clean=X_clean, 
                X_adv=X_adv, 
                y_clean=y_clean, 
                y_adv=y_adv, 
                num_classes=num_classes, 
                ignore_index=ignore_index
            )
        
        elif adv_mode == 'at_100':
            X_adv = attacker.attack(model, X, y)
            loss = adversarial_training_100_loss(
                model=model, 
                X_adv=X_adv, 
                y=y,
                num_classes=num_classes, 
                ignore_index=ignore_index
            )
            
        else:
            raise ValueError(f"Invalid training flag: {adv_mode}")
                         
        loss.backward()
        optimizer.step()
        scheduler.step()
           
        epoch_loss += loss.item() * X.size(0)
        num_samples += X.size(0)
            
        if is_main_process() and batch_idx % 20 == 0 and batch_idx != 0:
            current_lr = optimizer.param_groups[0]["lr"]
            dist_print(
                f"\tBatch [{batch_idx}/{len(dataloader)}] "
                f"lr={current_lr:.6e} "
                f"loss={loss.item():.4f}"
            )
          
    total_epoch_loss = reduce_sum(epoch_loss, device)
    total_num_samples = reduce_sum(num_samples, device)
    avg_loss = total_epoch_loss / max(total_num_samples, 1)
    
    return avg_loss         
    
    
def train(
    model,
    device,
    train_dataloader, 
    train_sampler,
    val_dataloader,
    optimizer, 
    scheduler, 
    num_epochs,
    save_dir, 
    num_classes,
    ignore_index,
    model_name,
    dataset_name,
    metric='iou',
    start_epoch=0,
    attacker=None,
    eval_attacker=None,
    adv_train=False,
    adv_weight=1.0,
    adv_mode=None,
    eval_interval=5,
    evaluate_robust=True
):
    if is_main_process():
        dist_print('Entered training loop')
        checkpoint_train_mode = ""
        if adv_mode is not None:
            checkpoint_train_mode = adv_mode
        else:
            if adv_train:
                checkpoint_train_mode = "at"
            else:
                checkpoint_train_mode = "clean"
        logger = MetricsLogger(
            os.path.join(save_dir, f'{model_name}_{dataset_name}_{checkpoint_train_mode}.csv'), 
            metric_name=metric, 
            num_classes=num_classes, 
            log_weights=False, 
            log_adv=evaluate_robust
        )
        eval_logger = EvalLogger(os.path.join(save_dir, f'{model_name}_{dataset_name}_{checkpoint_train_mode}_clean_metrics.csv'))
        eval_logger_adv = EvalLogger(os.path.join(save_dir, f'{model_name}_{dataset_name}_{checkpoint_train_mode}_robust_metrics.csv'))
    else:
        logger = None
        eval_logger = None
        eval_logger_adv = None
        checkpoint_train_mode = None
        
    previous_clean_scores = None
    previous_robust_scores = None
    
    try:
        best_score = 0.0
        mean_score = 0.0
        end_epoch = num_epochs
        for epoch in range(start_epoch, end_epoch):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            if is_main_process():
                dist_print(f'Epoch [{epoch+1} / {end_epoch}]')
                epoch_start_time = time.time()
            
            model.train()   

            avg_loss = train_one_epoch(
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                num_classes=num_classes,
                ignore_index=ignore_index,
                attacker=attacker,
                adv_train=adv_train,
                adv_mode=adv_mode,
                adv_weight=adv_weight
            )
            
            if is_main_process():
                train_time = time.time() - epoch_start_time
                dist_print(f'Finished training after {train_time:.2f} seconds')

            if (epoch % eval_interval) == 0:
                eval_results = evaluate(
                    model=model, 
                    dataloader=val_dataloader,
                    metric=metric, 
                    num_classes=num_classes, 
                    ignore_index=ignore_index, 
                    eval_logger=eval_logger,
                    model_name=model_name,
                    device=device
                )
                
                mean_score_clean = float(eval_results["mean_score"])
                class_scores_clean = eval_results['per_class_score']
                
                class_scores_adv = None
                mean_score_adv = None
                if evaluate_robust:
                    eval_results_adv = evaluate(
                        model=model, 
                        dataloader=val_dataloader, 
                        metric=metric, 
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                        attacker=eval_attacker,
                        eval_logger=eval_logger_adv,
                        model_name=model_name,
                        device=device
                    )
                    class_scores_adv = eval_results_adv['per_class_score']
                    mean_score_adv = float(eval_results_adv["mean_score"])
            
            if is_main_process():
                epoch_time = time.time() - epoch_start_time
                dist_print(f'Epoch time: {epoch_time:.2f} seconds')
                
                if (epoch % eval_interval) == 0:
                    dist_print(f"Avg. loss: {avg_loss}, mIoU: {mean_score_clean} robust mIoU: {mean_score_adv}")
                    
                    print_per_class_evolution(
                        epoch=epoch,
                        clean_scores=class_scores_clean,
                        robust_scores=class_scores_adv,
                        focus_classes=None,
                        previous_clean=previous_clean_scores,
                        previous_robust=previous_robust_scores,
                    )
                
                    logger.log(
                        epoch_idx=epoch,
                        avg_loss=avg_loss,
                        mean_score=mean_score,
                        mean_score_robust=mean_score_adv,
                        class_scores=class_scores_clean,
                        class_scores_robust=class_scores_adv
                    )
                    
                    save_checkpoint(
                        save_name=f'{model_name}_{dataset_name}_{checkpoint_train_mode}_latest',
                        save_dir=save_dir,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        scheduler=scheduler
                    )
                
                if mean_score_clean > best_score:
                    save_checkpoint(
                        save_name=f'{model_name}_{dataset_name}_{checkpoint_train_mode}_best',
                        save_dir=save_dir,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        scheduler=scheduler
                    )
                    best_score = mean_score
    finally:
        if logger is not None:
            logger.close()
