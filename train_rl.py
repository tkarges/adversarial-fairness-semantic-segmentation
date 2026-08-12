import argparse
import os
import time
from attacks.segpgd import SegPGD
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.ddp_utils import setup_ddp, dist_print, is_main_process, reduce_sum
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from datasets.metadata import get_dataset_info
import random
import numpy as np
from utils.logging import MetricsLogger, RLDecisionLogger
from evaluate import evaluate
from utils.stats import print_per_class_evolution, print_tail_summary
from attacks.pgd import TradesPGD
from rl_agent import (
    WeightPolicy,
    RewardBaseline,
    build_policy_state,
    action_to_class_weights,
    calculate_policy_reward,
    collect_class_frequencies,
    broadcast_class_weights,
    save_rl_checkpoint,
    load_rl_checkpoint,
    select_fixed_bottom_classes,
    load_training_class_frequencies
)
from losses_rl import adversarial_training_loss

def calculate_disparity(scores):
    scores = torch.as_tensor(
        scores,
        dtype=torch.float32,
    )

    valid = scores[torch.isfinite(scores)]
    sorted_scores = torch.sort(valid).values
    lower_half_count = max(1, valid.numel() // 2)

    return {
        "min": sorted_scores[0].item(),
        "worst3": sorted_scores[:3].mean().item(),
        "worst5": sorted_scores[:5].mean().item(),
        "lower_half": sorted_scores[
            :lower_half_count
        ].mean().item(),
        "std": valid.std(unbiased=False).item(),
        "range": (
            valid.max() - valid.min()
        ).item(),
    }


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    num_classes,
    ignore_index,
    device,
    attacker=None,
    adv_train=False,
    use_clean_weights=False,
    use_adv_weights=True,
    clean_weight_mode="pixel",
    adv_weight_mode="pixel",
    rl_class_weights=None,
):
    epoch_loss = 0.0
    num_samples = 0
    
    for batch_idx, (X, y) in enumerate(dataloader):
        X = X.to(device, non_blocking=device.type == 'cuda')
        y = y.to(device, non_blocking=device.type == 'cuda')
        
        optimizer.zero_grad(set_to_none=True)

        batch_size = X.size(0)
        perm = torch.randperm(batch_size, device=X.device)

        adv_size = batch_size // 2
        adv_idx = perm[:adv_size]
        clean_idx = perm[adv_size:]

        X_adv_base = X[adv_idx]
        y_adv = y[adv_idx]

        X_clean = X[clean_idx]
        y_clean = y[clean_idx]

        X_adv = attacker.attack(
            model=model,
            X=X_adv_base,
            mask=y_adv,
        )

        loss = adversarial_training_loss(
            model=model,
            X_clean=X_clean,
            X_adv=X_adv,
            y_clean=y_clean,
            y_adv=y_adv,
            num_classes=num_classes,
            ignore_index=ignore_index,
            class_weights=rl_class_weights if use_clean_weights else None,
            class_weights_adv=rl_class_weights if use_adv_weights else None,
            clean_weight_mode=clean_weight_mode,
            adv_weight_mode=adv_weight_mode
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
        
def train_rl(
    model,
    device,
    train_dataloader, 
    train_sampler,
    val_dataloader,
    policy_dataloader,
    policy_split_path,
    policy_hists_path,
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
    adv_train=True,
    eval_interval=5,
    evaluate_robust=True,
    distributed=False,
    use_clean_weights=False,
    use_adv_weights=True,
    clean_weight_mode="pixel",
    adv_weight_mode="pixel",
    rl_action_epochs=2,
    rl_hidden_dim=32,
    rl_learning_rate=1e-4,
    rl_entropy_coefficient=1e-3,
    rl_reward_scale=10.0,
    rl_uniform_mix=0.5,
    rl_min_weight=0.5,
    rl_max_weight=2.0,
    rl_checkpoint_path=None,
):
    if is_main_process():
        weight_application = "disabled"
        if use_clean_weights and use_adv_weights:
            weight_application = "both"
        elif use_clean_weights and not use_adv_weights:
            weight_application = "clean"
        elif not use_clean_weights and use_adv_weights:
            weight_application = "adv"
        checkpoint_name = f"RL_{model_name}_{dataset_name}_weighted_{weight_application}_mode_{clean_weight_mode}_{adv_weight_mode}"
        logger = MetricsLogger(
            os.path.join(save_dir, f"{checkpoint_name}_metrics.csv"), 
            metric_name=metric, 
            num_classes=num_classes, 
            log_weights=False, 
            log_adv=True
        )
        rl_logger = RLDecisionLogger(
            path=os.path.join(save_dir, f"{checkpoint_name}_rl_decisions.csv"),
            num_classes=num_classes,
            action_dim=4,
        )
    else:
        logger = None
        rl_logger = None
        checkpoint_name = None
        weight_application = None
    
    dist_print(
        "Effective objective: "
        f"clean_mode={clean_weight_mode}, "
        f"clean_weighted={use_clean_weights}, "
        f"adv_mode={adv_weight_mode}, "
        f"adv_weighted={use_adv_weights}"
    )
    
    previous_clean_scores = None
    previous_robust_scores = None 

    rl_policy = None
    rl_optimizer = None
    rl_baseline = None
    
    rl_class_weights = torch.ones(num_classes, device=device, dtype=torch.float32)
    
    # Important for trainig because probability and entropy of an action need to be stored until the reward is available
    rl_pending_log_prob = None
    rl_pending_entropy = None
    
    # Only relevant for logging, could be deleted
    rl_interval_start_clean = None
    rl_interval_start_robust = None
    rl_previous_robust = None
    rl_class_frequencies = None
    rl_fixed_bottom_classes = None
    rl_pending_policy_diag = None
    rl_pending_weight_diag = None
    rl_pending_candidate_weights = None
    rl_pending_active_weights = None
    rl_pending_clean_scores = None
    rl_pending_robust_scores = None
    rl_pending_epoch = None
    rl_decision_step = 0

    if not evaluate_robust:
        raise ValueError("RL-based training requires --evaluate-robust.")
    if eval_interval != rl_action_epochs:
        raise ValueError(
            "For RL-based training, --eval-interval must equal --rl-action-epochs so every action receives one reward."
        )
    '''
    rl_class_frequencies = collect_class_frequencies(
        dataloader=train_dataloader,
        num_classes=num_classes,
        ignore_index=ignore_index,
        device=device,
    )
    '''
    rl_class_frequencies = load_training_class_frequencies(
        histogram_path=policy_hists_path,
        split_idx_path=policy_split_path,
        device=device
    )
    if is_main_process():
        # Initialize policy, optimizer and baseline
        state_dim = num_classes * 4 + 5
        rl_policy = WeightPolicy(
            state_dim=state_dim,
            hidden_dim=rl_hidden_dim,
        ).to(device)
        rl_optimizer = torch.optim.Adam(
            rl_policy.parameters(),
            lr=rl_learning_rate,
        )
        rl_baseline = RewardBaseline(momentum=0.9)

        # Checkpoint can be loaded if specified in args
        if rl_checkpoint_path is not None:
            rl_class_weights, rl_previous_robust, rl_fixed_bottom_classes = load_rl_checkpoint(
                path=rl_checkpoint_path,
                policy=rl_policy,
                optimizer=rl_optimizer,
                baseline=rl_baseline,
                device=device,
            )

    # Class weights should be the same on every GPU
    rl_class_weights = broadcast_class_weights(
        weights=rl_class_weights if is_main_process() else None,
        num_classes=num_classes,
        device=device,
    ).detach()

    rl_weights_used = None

    try:
        best_score = 0.0
        mean_score = 0.0
        end_epoch = num_epochs
        for epoch in range(start_epoch, num_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            if is_main_process():
                dist_print(f'Epoch [{epoch+1} / {num_epochs}]')
                epoch_start_time = time.time()
            
            model.train()   

            avg_loss = train_one_epoch(
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                num_classes=num_classes,
                ignore_index=ignore_index,
                device=device,
                attacker=attacker,
                adv_train=adv_train,
                use_clean_weights=use_clean_weights,
                use_adv_weights=use_adv_weights,
                clean_weight_mode=clean_weight_mode,
                adv_weight_mode=adv_weight_mode,
                rl_class_weights=rl_class_weights,
            )    
            
            if is_main_process():
                train_time = time.time() - epoch_start_time
                dist_print(f'Finished training after {train_time:.2f} seconds')

            if (epoch % eval_interval) == 0:
                eval_results = evaluate(
                    model=model, 
                    dataloader=policy_dataloader,
                    metric=metric, 
                    num_classes=num_classes, 
                    ignore_index=ignore_index,
                    device=device
                )
                mean_score_clean = float(eval_results["mean_score"])
                class_scores_clean = eval_results['per_class_score']
                
                scores = torch.as_tensor(class_scores_clean, dtype=torch.float32)

                valid_scores = scores[torch.isfinite(scores)]
                
                class_scores_adv = None
                mean_score_adv = None
                if evaluate_robust:
                    eval_results_adv = evaluate(
                        model=model, 
                        dataloader=policy_dataloader, 
                        metric=metric, 
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                        attacker=eval_attacker,
                        device=device
                    )
                    class_scores_adv = eval_results_adv['per_class_score']
                    mean_score_adv = float(eval_results_adv["mean_score"])

                if class_scores_adv is None:
                    raise RuntimeError("RL weights require robust per-class scores.")

                if is_main_process():
                    current_clean = torch.nan_to_num(
                        torch.as_tensor(
                            class_scores_clean,
                            dtype=torch.float32,
                            device=device,
                        ),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    current_robust = torch.nan_to_num(
                        torch.as_tensor(
                            class_scores_adv,
                            dtype=torch.float32,
                            device=device,
                        ),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    
                    if rl_interval_start_robust is not None and rl_weights_used is not None:
                        robust_delta = current_robust - rl_interval_start_robust


                        dist_print("\nRL interval per-class diagnostics:")

                        for class_idx in range(num_classes):
                            dist_print(
                                f"  {class_idx:02d} "
                                f"weight={rl_weights_used[class_idx].item():.4f} "
                                f"old_robust={rl_interval_start_robust[class_idx].item():.4f} "
                                f"new_robust={current_robust[class_idx].item():.4f} "
                                f"delta={robust_delta[class_idx].item():+.4f}"
                            )
                        
                    if is_main_process() and rl_fixed_bottom_classes is None:
                        rl_fixed_bottom_classes = select_fixed_bottom_classes(
                            baseline_robust_scores=current_robust,
                            k=5,
                        )

                        dist_print(
                            "Fixed baseline bottom-5 classes: "
                            f"{rl_fixed_bottom_classes.detach().cpu().tolist()}"
                        )

                        dist_print(
                            "Fixed baseline bottom-5 robust scores: "
                            f"{current_robust[rl_fixed_bottom_classes].detach().cpu()}"
                        )

                    # If RL has already taken an action, the reward can be computed
                    # This is skipped for the first action, as there is no observed effect yet
                    if rl_pending_log_prob is not None and rl_interval_start_clean is not None and rl_interval_start_robust is not None:
                        reward, reward_diag = calculate_policy_reward(
                            old_clean=rl_interval_start_clean,
                            old_robust=rl_interval_start_robust,
                            new_clean=current_clean,
                            new_robust=current_robust,
                            fixed_target_classes=rl_fixed_bottom_classes,
                            dynamic_bottom_k=5,
                            robust_drop_tolerance=0.005,
                            clean_drop_tolerance=0.015,
                        )
                        scaled_reward = rl_reward_scale * reward
                        baseline_value = rl_baseline.get()
                        advantage = scaled_reward - baseline_value
                        
                        # REINFORCE style policy gradient loss with entropy regularization
                        policy_loss = -advantage * rl_pending_log_prob - rl_entropy_coefficient * rl_pending_entropy

                        # Update the policy
                        rl_optimizer.zero_grad(set_to_none=True)
                        policy_loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            rl_policy.parameters(),
                            max_norm=1.0,
                        )
                        rl_optimizer.step()
                        rl_baseline.update(scaled_reward)
                        
                        if (
                            rl_logger is not None
                            and rl_pending_policy_diag is not None
                            and rl_pending_weight_diag is not None
                            and rl_pending_candidate_weights is not None
                            and rl_pending_active_weights is not None
                        ):  
                            rl_logger.log(
                                decision_step=rl_decision_step,
                                epoch=rl_pending_epoch,
                                policy_diagnostics=rl_pending_policy_diag,
                                weight_diagnostics=rl_pending_weight_diag,
                                candidate_weights=rl_pending_candidate_weights,
                                active_weights=rl_pending_active_weights,
                                clean_scores=rl_pending_clean_scores,
                                robust_scores=rl_pending_robust_scores,
                                reward=float(reward),
                                scaled_reward=float(scaled_reward),
                                baseline_before=float(baseline_value),
                                advantage=float(advantage),
                                policy_loss=float(policy_loss.detach().item()),
                                grad_norm=float(grad_norm),
                            )

                            rl_decision_step += 1

                        dist_print(
                            "RL update: "
                            f"reward={reward:+.6f} "
                            f"advantage={advantage:+.4f} "
                            f"policy_loss={policy_loss.item():+.4f} "
                            f"grad_norm={float(grad_norm):.4f} "
                            f"components={reward_diag}"
                        )

                    previous_for_state = current_robust if rl_previous_robust is None else rl_previous_robust
                    
                    # Builds the state for RL
                    state = build_policy_state(
                        clean_scores=current_clean,
                        robust_scores=current_robust,
                        previous_robust_scores=previous_for_state,
                        epoch=epoch,
                        end_epoch=end_epoch,
                    )
                    
                    # Samples an action fromt that state
                    action, rl_pending_log_prob, rl_pending_entropy, rl_policy_diagnostics = rl_policy.sample(state)
                    
                    # Computes class weights from the action
                    sampled_weights, rl_diag = action_to_class_weights(
                        action=action,
                        clean_scores=current_clean,
                        robust_scores=current_robust,
                        class_frequencies=rl_class_frequencies,
                        fixed_target_classes=rl_fixed_bottom_classes,
                        uniform_mix=rl_uniform_mix,
                        min_weight=rl_min_weight,
                        max_weight=rl_max_weight,
                    )
                    
                    rl_pending_policy_diag = {
                        key: (
                            value.detach().cpu().clone()
                            if torch.is_tensor(value)
                            else value
                        )
                        for key, value in rl_policy_diagnostics.items()
                    }

                    rl_pending_weight_diag = {
                        key: (
                            value.detach().cpu().clone()
                            if torch.is_tensor(value)
                            else value
                        )
                        for key, value in rl_diag.items()
                    }
                    
                    rl_pending_candidate_weights = (
                        sampled_weights.detach().cpu().clone()
                    )

                    rl_pending_active_weights = (
                        sampled_weights.detach().cpu().clone()
                    )

                    rl_pending_clean_scores = (
                        current_clean.detach().cpu().clone()
                    )
                    rl_pending_robust_scores = (
                        current_robust.detach().cpu().clone()
                    )
                    rl_pending_epoch = int(epoch)
                    
                    dist_print(f"active_weights={sampled_weights.detach().cpu()}")
                    rl_interval_start_clean = current_clean.detach().clone()
                    rl_interval_start_robust = current_robust.detach().clone()
                    rl_previous_robust = current_robust.detach().clone()

                    coefficients = rl_diag["coefficients"].detach().cpu()

                    dist_print(
                        "RL action: "
                        f"weakness={coefficients[0].item():.4f} "
                        f"gap={coefficients[1].item():.4f} "
                        f"rarity={coefficients[2].item():.4f} "
                        f"target={coefficients[3].item():.4f} "
                        f"target_weights="
                        f"{sampled_weights[rl_fixed_bottom_classes].detach().cpu()} "
                        f"weights={sampled_weights.detach().cpu()}"
                    )
                
                else:
                    sampled_weights = None

                rl_class_weights = broadcast_class_weights(
                    weights=sampled_weights,
                    num_classes=num_classes,
                    device=device,
                ).detach()

                rl_weights_used = rl_class_weights.detach().clone()
            
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
                        previous_clean=previous_clean_scores,
                        previous_robust=previous_robust_scores,
                    )
                    
                    clean_disparity = calculate_disparity(
                        class_scores_clean
                    )

                    robust_disparity = calculate_disparity(
                        class_scores_adv
                    )
                    
                    dist_print(
                        "Clean disparity: "
                        f"min={clean_disparity['min']:.4f} "
                        f"worst3={clean_disparity['worst3']:.4f} "
                        f"worst5={clean_disparity['worst5']:.4f} "
                        f"lower_half={clean_disparity['lower_half']:.4f} "
                        f"std={clean_disparity['std']:.4f} "
                        f"range={clean_disparity['range']:.4f}"
                    )

                    dist_print(
                        "Robust disparity: "
                        f"min={robust_disparity['min']:.4f} "
                        f"worst3={robust_disparity['worst3']:.4f} "
                        f"worst5={robust_disparity['worst5']:.4f} "
                        f"lower_half={robust_disparity['lower_half']:.4f} "
                        f"std={robust_disparity['std']:.4f} "
                        f"range={robust_disparity['range']:.4f}"
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

                    save_rl_checkpoint(
                        path=os.path.join(save_dir, f"{checkpoint_name}_rl.pt"),
                        policy=rl_policy,
                        optimizer=rl_optimizer,
                        baseline=rl_baseline,
                        weights=rl_class_weights,
                        previous_robust=rl_previous_robust,
                        rl_fixed_bottom_classes=rl_fixed_bottom_classes
                    )

                    # Early stopping
                    selection_score = current_robust[rl_fixed_bottom_classes].mean().item()

                    if selection_score > best_score:
                        save_checkpoint(
                            save_name=f"{checkpoint_name}_best",
                            save_dir=save_dir,
                            model=model,
                            optimizer=optimizer,
                            epoch=epoch,
                            scheduler=scheduler,
                        )

                        save_rl_checkpoint(
                            path=os.path.join(save_dir, f"{checkpoint_name}_best_rl.pt"),
                            policy=rl_policy,
                            optimizer=rl_optimizer,
                            baseline=rl_baseline,
                            weights=rl_class_weights,
                            previous_robust=rl_previous_robust,
                            rl_fixed_bottom_classes=rl_fixed_bottom_classes
                        )

                        best_score = selection_score
                        
            if distributed:
                torch.distributed.barrier()
                
        if is_main_process():        
            save_checkpoint(
                save_name=f"{checkpoint_name}_last",
                save_dir=save_dir,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                scheduler=scheduler
            )
            
            save_rl_checkpoint(
                path=os.path.join(save_dir, f"{checkpoint_name}_last_rl.pt"),
                policy=rl_policy,
                optimizer=rl_optimizer,
                baseline=rl_baseline,
                weights=rl_class_weights,
                previous_robust=rl_previous_robust,
                rl_fixed_bottom_classes=rl_fixed_bottom_classes
            )
        
    finally:
        if logger is not None:
            logger.close()
        if rl_logger is not None:
            rl_logger.close()