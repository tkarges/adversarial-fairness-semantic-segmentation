'''
DAFA Training Script
'''
import os
import time
from losses_dafa import dafa_dense_loss, dafa_trades_loss, compute_dafa_weights, collect_dafa_statistics
from utils.checkpoint import save_checkpoint
from utils.ddp_utils import dist_print, is_main_process, reduce_sum
import torch
from utils.logging import MetricsLogger
from evaluate import evaluate
from utils.stats import print_per_class_evolution, print_tail_summary


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    num_classes,
    ignore_index,
    device,
    attacker=None,
    adv_weight=1.0,
    dafa_mode="segpgd",
    dafa_beta=1.0, 
    dafa_weights=None,
    dafa_weights_active=False
):
    epoch_loss = 0.0
    num_samples = 0
    
    for batch_idx, (X, y) in enumerate(dataloader):
        X = X.to(device, non_blocking=device.type == 'cuda')
        y = y.to(device, non_blocking=device.type == 'cuda')
        
        optimizer.zero_grad(set_to_none=True)
        
        # Use DAFA weights if they are currently active (if warmup is over)
        active_weights = dafa_weights if dafa_weights_active else None

        if dafa_mode == "trades-pgd":
            # DAFA-TRADES loss
            loss = dafa_trades_loss(
                model=model,
                X=X,
                y=y,
                attacker=attacker,
                num_classes=num_classes,
                ignore_index=ignore_index,
                class_weights=active_weights,
                beta=dafa_beta,
            )
        
        elif dafa_mode == "segpgd":
            # SegPGD-AT batch split
            batch_size = X.size(0)
            perm = torch.randperm(batch_size, device=X.device)

            adv_size = batch_size // 2
            adv_idx = perm[:adv_size]
            clean_idx = perm[adv_size:]

            X_adv_base = X[adv_idx]
            y_adv = y[adv_idx]

            X_clean = X[clean_idx]
            y_clean = y[clean_idx]

            # DAFA-CE loss
            loss = dafa_dense_loss(
                model=model,
                attacker=attacker,
                X_clean=X_clean,
                X_adv_base=X_adv_base,
                y_clean=y_clean,
                y_adv=y_adv,
                num_classes=num_classes,
                ignore_index=ignore_index,
                class_weights=active_weights,
                adv_weight=adv_weight
            )

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
    
    
def train_dafa(
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
    adv_weight=1.0,
    dafa_mode="segpgd",
    dafa_warmup_epochs=70,
    dafa_lambda=1.0,
    dafa_beta=1.0,
    eval_interval=5,
    evaluate_robust=True,
    distributed=False,
):
    if is_main_process():
        checkpoint_name = f"{model_name}_{dataset_name}_dafa_{dafa_mode}"
        logger = MetricsLogger(
            os.path.join(save_dir, f"{checkpoint_name}_metrics.csv"), 
            metric_name=metric, 
            num_classes=num_classes, 
            log_weights=False, 
            log_adv=False
        )
    else:
        logger = None
        checkpoint_name = None
    
    previous_clean_scores = None
    previous_robust_scores = None

    # Indicates whether class weights should be used in the current epoch
    dafa_weights_active = False
    
    # Aggregation of class-wise probabilities will be saved in this variable
    dafa_prob_sum = torch.zeros(num_classes, num_classes, dtype=torch.float64, device=device)

    # Aggregation of class counts is additionally required in semantic segmentation
    dafa_class_counts = torch.zeros(num_classes, dtype=torch.float64, device=device)

    dafa_weights = torch.ones(num_classes, dtype=torch.float32, device=device)
        
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
            
            # DAFA stats for weight computation are computed in the warmup epoch
            dafa_stats_epoch = (epoch == dafa_warmup_epochs - 1)

            # Weights are only relevant when warmup is done
            dafa_weights_active = (epoch >= dafa_warmup_epochs)
            
            if dafa_stats_epoch:
                dafa_prob_sum.zero_()
                dafa_class_counts.zero_()
            
            model.train()   

            # Trains the model for one epoch
            avg_loss = train_one_epoch(
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                num_classes=num_classes,
                ignore_index=ignore_index,
                device=device,
                attacker=attacker,
                adv_weight=adv_weight,
                dafa_mode=dafa_mode,
                dafa_beta=dafa_beta,
                dafa_weights=dafa_weights,
                dafa_weights_active=dafa_weights_active
            )
            
            # This condition is true immediately after warmup --> weights are now computed
            # This is only done once! (see thesis for why this is a limitation)
            if dafa_stats_epoch:
                if distributed:
                    torch.distributed.barrier()

                # Save warmup checkpoint for convenience
                if is_main_process():
                    save_checkpoint(
                        save_name=f"dafa_precompute_epoch_{epoch + 1}",
                        save_dir=save_dir,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        scheduler=scheduler,
                    )

                    dist_print(
                        "Saved pre-DAFA checkpoint before statistics "
                        f"computation at epoch {epoch + 1}"
                    )

                if distributed:
                    torch.distributed.barrier()

                # Stats are computed to construct the C x C probability matrix
                collect_dafa_statistics(
                    model=model,
                    dataloader=train_dataloader,
                    attacker=attacker,
                    prob_sum=dafa_prob_sum,
                    class_counts=dafa_class_counts,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    device=device,
                )

                # Multiple devices need to be taken into account
                if distributed:
                    torch.distributed.all_reduce(dafa_prob_sum)
                    torch.distributed.all_reduce(dafa_class_counts)

                # Initialize the already declared weights variable with the computed class weights
                dafa_weights.copy_(
                    compute_dafa_weights(
                        prob_sum=dafa_prob_sum,
                        class_counts=dafa_class_counts,
                        dafa_lambda=dafa_lambda,
                    )
                )

                if is_main_process():
                    dist_print(
                        f"Computed DAFA weights after epoch {epoch + 1}\n"
                        f"DAFA weights: {dafa_weights.detach().cpu()}"
                    )
            
            if is_main_process():
                train_time = time.time() - epoch_start_time
                dist_print(f'Finished training after {train_time:.2f} seconds')

            # Evaluation
            if (epoch % eval_interval) == 0:
                # Clean evaluation
                eval_results = evaluate(
                    model=model, 
                    dataloader=val_dataloader,
                    metric=metric, 
                    num_classes=num_classes, 
                    ignore_index=ignore_index,
                    device=device
                )
                
                # -----------------------------------------------------
                # This is just for console output
                mean_score_clean = float(eval_results["mean_score"])
                class_scores_clean = eval_results['per_class_score']
                
                scores = torch.as_tensor(class_scores_clean, dtype=torch.float32)

                valid_scores = scores[torch.isfinite(scores)]
                sorted_scores = torch.sort(valid_scores).values

                min_iou = sorted_scores[0].item()
                worst3_iou = sorted_scores[:3].mean().item()
                worst5_iou = sorted_scores[:5].mean().item()

                lower_half_count = max(
                    1,
                    sorted_scores.numel() // 2,
                )

                lower_half_iou = (
                    sorted_scores[:lower_half_count]
                    .mean()
                    .item()
                )

                class_std = (
                    valid_scores.std(unbiased=False).item()
                )

                class_range = (
                    valid_scores.max()
                    - valid_scores.min()
                ).item()
                # -----------------------------------------------------
                
                class_scores_adv = None
                mean_score_adv = None
                if evaluate_robust:
                    # Robust evaluation
                    eval_results_adv = evaluate(
                        model=model, 
                        dataloader=val_dataloader, 
                        metric=metric, 
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                        attacker=eval_attacker,
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

                    print_tail_summary(
                        clean_scores=class_scores_clean,
                        robust_scores=class_scores_adv,
                        dataset_name=dataset_name,
                        previous_clean=previous_clean_scores,
                        previous_robust=previous_robust_scores,
                    )
                    
                    dist_print(
                        "Disparity metrics: "
                        f"min={min_iou:.4f} "
                        f"worst3={worst3_iou:.4f} "
                        f"worst5={worst5_iou:.4f} "
                        f"lower_half={lower_half_iou:.4f} "
                        f"std={class_std:.4f} "
                        f"range={class_range:.4f}"
                    )

                    previous_clean_scores = torch.as_tensor(class_scores_clean, dtype=torch.float32).detach().cpu()
                    
                    if class_scores_adv is not None:
                        previous_robust_scores = torch.as_tensor(class_scores_adv, dtype=torch.float32).detach().cpu()
                
                    logger.log(
                        epoch_idx=epoch,
                        avg_loss=avg_loss,
                        mean_score=mean_score_clean,
                        mean_score_robust=mean_score_adv,
                        class_scores=class_scores_clean,
                        class_scores_robust=class_scores_adv if class_scores_adv is not None else None
                    )
                    
                    save_checkpoint(
                        save_name=checkpoint_name,
                        save_dir=save_dir,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        scheduler=scheduler
                    )               
                
                if mean_score_clean > best_score:
                    save_checkpoint(
                        save_name=f'{checkpoint_name}_best',
                        save_dir=save_dir,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        scheduler=scheduler
                    )
                    best_score = mean_score_clean
                    
        save_checkpoint(
            save_name=f"{checkpoint_name}_last",
            save_dir=save_dir,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            scheduler=scheduler
        )
    finally:
        if logger is not None:
            logger.close()