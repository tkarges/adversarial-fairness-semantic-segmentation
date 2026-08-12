import argparse
import numpy as np
import random
import os
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from attacks.segpgd import SegPGD
from attacks.pgd import TradesPGD
from utils.checkpoint import load_checkpoint
from utils.ddp_utils import setup_ddp, destroy_process_group, dist_print
from datasets.loading import build_dense_dataloaders
from datasets.metadata import get_dataset_info, get_policy_split_path
from models.load_model import load_dense_model
from train_normal import train
from train_dafa import train_dafa
from train_rl import train_rl

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def build_poly_scheduler(optimizer, num_epochs, steps_per_epoch, power=0.9):
    total_steps = num_epochs * steps_per_epoch
    
    def lr_lambda(current_step):
        if current_step >= total_steps:
            return 0.0
        return (1 - current_step / total_steps) ** power
    
    return LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)

def parse_args():
    parser = argparse.ArgumentParser()
    # Specifies which training script is invoked
    parser.add_argument("--training-type", type=str, choices=["normal", "dafa", "rl"])
    
    parser.add_argument("--model", type=str, choices=["deeplabv3", "pspnet", "mask2former"])
    parser.add_argument("--dataset", type=str, choices=["cityscapes", "voc2012"])
    
    parser.add_argument("--model-savedir", type=str, default=".")
    parser.add_argument("--load-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-path", type=str)
    parser.add_argument("--checkpoint-finetune", action="store_true")
    
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--evaluate-robust", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=10)
    
    parser.add_argument("--adversarial-training", action="store_true")
    parser.add_argument("--adversarial-loss-weight", type=float, default=1.0)
    parser.add_argument("--adversarial-training-mode", type=str, default=None)
    
    parser.add_argument("--prob-class-crop", type=float, default=0.0)
    
    # The attacker that is used for adversarial training (trades-pgd is only available for DAFA)
    parser.add_argument("--attacker", type=str, default="segpgd")
    parser.add_argument("--attack-iterations", type=int, default=3)
    parser.add_argument("--attack-epsilon", type=float, default=8.0/255)
    parser.add_argument("--attack-alpha", type=float, default=2.0/255)
    
    # These values should not be changed to guarantee evaluation under a common threat model
    parser.add_argument("--eval-attacker", type=str, choices=["segpgd"])
    parser.add_argument("--eval-attack-iterations", type=int, default=3)
    parser.add_argument("--eval-attack-epsilon", type=float, default=8.0/255)
    parser.add_argument("--eval-attack-alpha", type=float, default=2.0/255)
    
    parser.add_argument("--dafa-mode", type=str, choices=["segpgd", "trades-pgd"])
    parser.add_argument("--dafa-warmup-epochs", type=int, default=70)
    parser.add_argument("--dafa-lambda", type=float, default=1.0)
    parser.add_argument("--dafa-beta", type=float, default=1.0)
    parser.add_argument("--dafa-scale-margins", action="store_true")
    
    parser.add_argument("--rl-action-epochs", type=int, default=2)
    parser.add_argument("--rl-hidden-dim", type=int, default=32)
    parser.add_argument("--rl-learning-rate", type=float, default=1e-4)
    parser.add_argument("--rl-entropy-coefficient", type=float, default=1e-3)
    parser.add_argument("--rl-reward-scale", type=float, default=10.0)
    parser.add_argument("--rl-uniform-mix", type=float, default=0.50)
    parser.add_argument("--rl-min-weight", type=float, default=1.0)
    parser.add_argument("--rl-max-weight", type=float, default=1.4)
    parser.add_argument("--rl-checkpoint-path", type=str, default=None)
    parser.add_argument("--rl-apply-clean-weight", action="store_true")
    parser.add_argument("--rl-apply-adv-weight", action="store_true")
    parser.add_argument("--rl-clean-weight-mode", type=str, default="pixel")
    parser.add_argument("--rl-adv-weight-mode", type=str, default="pixel")
    
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()
    
def validate_args(args):
    valid = True
    if not valid:
        raise ValueError(f"Invalid arguments {args}")
    
def main():
    local_rank, device, distributed = setup_ddp()
    dist_print(f"environment setup completed with local_rank={local_rank}, device={device}, distributed={distributed}")
    dist_print(f"----------------------------")
    dist_print(f"Parsing and validating arguments")
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    os.makedirs(args.model_savedir, exist_ok=True)
    dist_print(f"Main script invoked with arguments {args}")
    dist_print(f"----------------------------")
    dist_print(f"Constructing model, dataloaders, optimizer, and scheduler")
    
    num_classes, ignore_index, data_root, _, _ = get_dataset_info(args.dataset)
    
    policy_split_path, policy_hists_path = None, None
    if args.training_type == "rl":
        policy_split_path, policy_hists_path = get_policy_split_path(args.dataset)
    if policy_split_path is None and args.training_type == "rl":
        raise ValueError(f"Reinforcement learning approach requires a valid policy dataset split saved in .pt format")
        
    train_loader, val_loader, train_sampler, val_sampler, policy_loader, policy_sampler = build_dense_dataloaders(
        dataset=args.dataset,
        data_root=data_root,
        distributed=distributed,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        prob_class_crop=args.prob_class_crop,
        use_rl=args.training_type == "rl",
        split_idx_path=policy_split_path
    )
    
    rank = torch.distributed.get_rank() if distributed else 0

    if policy_loader is None:
        print(
            f"[rank {rank}] "
            f"train batches={len(train_loader)}, "
            f"val batches={len(val_loader)}, "
            f"train sampler={type(train_sampler).__name__}, "
            f"val sampler={type(val_sampler).__name__}",
            flush=True
        )
    else:
        print(
            f"[rank {rank}] "
            f"train batches={len(train_loader)}, "
            f"val batches={len(val_loader)}, "
            f"train sampler={type(train_sampler).__name__}, "
            f"val sampler={type(val_sampler).__name__} "
            f"policy evaluation batches={len(policy_loader)} "
            f"policy sampler={type(policy_sampler).__name__}",
            flush=True
        )
        Xp, yp = next(iter(policy_loader))
        Xt, yt = next(iter(train_loader))
        print(
            f"[rank {rank}] "
            f"train batch: X={Xt.shape}, y={yt.shape}, "
            f"train batch: X={Xp.shape}, y={yp.shape}",
            flush=True
        )
                  
    model = load_dense_model(
        model_name=args.model,
        num_classes=num_classes,
        device=device,
        distributed=distributed,
        load_existing=args.load_checkpoint
    ) 
    
    local_start_epoch = None
    local_num_epochs = None
    
    if args.load_checkpoint:
        if args.checkpoint_finetune:
            checkpoint_epoch = load_checkpoint(
                model=model,
                save_dir=args.checkpoint_path,
                optimizer=None,
                scheduler=None,
                device=device,
                load_optimizer=False,
                load_scheduler=False
            )
            
            optimizer = SGD(
                params=model.parameters(),
                lr=args.learning_rate,
                momentum=0.9,
                weight_decay=4e-5
            )
            
            scheduler = build_poly_scheduler(
                optimizer=optimizer,
                num_epochs=args.epochs,
                steps_per_epoch=len(train_loader),
                power=0.9
            )
            
            local_start_epoch = 0
            local_num_epochs = args.epochs
        
        else:
            optimizer = SGD(
                params=model.parameters(),
                lr=args.learning_rate,
                momentum=0.9,
                weight_decay=4e-5
            )
            
            scheduler = build_poly_scheduler(
                optimizer=optimizer,
                num_epochs=args.epochs,
                steps_per_epoch=len(train_loader),
                power=0.9
            )
            
            checkpoint_epoch = load_checkpoint(
                model=model,
                save_dir=args.checkpoint_path,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                load_optimizer=True,
                load_scheduler=True
            )
            
            local_start_epoch = checkpoint_epoch
            local_num_epochs = args.epochs
    
    else:
        optimizer = SGD(
            params=model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=4e-5
        )
            
        scheduler = build_poly_scheduler(
            optimizer=optimizer,
            num_epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            power=0.9
        )
        local_start_epoch = 0
        local_num_epochs = args.epochs
            
    
    if args.training_type == "normal" or args.training_type == "rl": 
        train_attacker = SegPGD(
            iterations=args.attack_iterations,
            epsilon=args.attack_epsilon,
            alpha=args.attack_alpha,
            num_classes=num_classes,
            ignore_index=ignore_index
        )
        
    elif args.training_type == "dafa":
        if args.dafa_mode == "trades-pgd":
            train_attacker = TradesPGD(
                iterations=args.attack_iterations,
                epsilon=args.attack_epsilon,
                alpha=args.attack_alpha,
                num_classes=num_classes,
                ignore_index=ignore_index,
                scale_margins=args.dafa_scale_margins
            )
        else:
            train_attacker = SegPGD(
                iterations=args.attack_iterations,
                epsilon=args.attack_epsilon,
                alpha=args.attack_alpha,
                num_classes=num_classes,
                ignore_index=ignore_index,
                scale_margins=args.dafa_scale_margins
            ) 
    
    eval_attacker = SegPGD(
        iterations=args.eval_attack_iterations,
        epsilon=args.eval_attack_epsilon,
        alpha=args.eval_attack_alpha,
        num_classes=num_classes,
        ignore_index=ignore_index
    )
    
    try:
        if args.training_type == "normal":
            train(
                model=model,
                device=device,
                train_dataloader=train_loader,
                train_sampler=train_sampler,
                val_dataloader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=local_num_epochs,
                save_dir=args.model_savedir,
                num_classes=num_classes,
                ignore_index=ignore_index,
                model_name=args.model,
                dataset_name=args.dataset,
                metric="iou",
                start_epoch=local_start_epoch,
                attacker=train_attacker,
                eval_attacker=eval_attacker,
                adv_train=args.adversarial_training,
                adv_weight=args.adversarial_loss_weight,
                adv_mode=args.adversarial_training_mode,
                eval_interval=args.eval_interval,
                evaluate_robust=args.evaluate_robust
            )
        elif args.training_type == "dafa":
            train_dafa(
                model=model,
                device=device,
                train_dataloader=train_loader,
                train_sampler=train_sampler,
                val_dataloader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=local_num_epochs,
                save_dir=args.model_savedir,
                num_classes=num_classes,
                ignore_index=ignore_index,
                model_name=args.model,
                dataset_name=args.dataset,
                metric="iou",
                start_epoch=local_start_epoch,
                attacker=train_attacker,
                eval_attacker=eval_attacker,
                adv_weight=args.adversarial_loss_weight,
                dafa_mode=args.dafa_mode,
                dafa_warmup_epochs=args.dafa_warmup_epochs,
                dafa_lambda=args.dafa_lambda,
                dafa_beta=args.dafa_beta,
                eval_interval=args.eval_interval,
                evaluate_robust=args.evaluate_robust,
                distributed=distributed
            )
        elif args.training_type == "rl":
            train_rl(
                model=model,
                device=device,
                train_dataloader=train_loader,
                train_sampler=train_sampler,
                val_dataloader=val_loader,
                policy_dataloader=policy_loader,
                policy_split_path=policy_split_path,
                policy_hists_path=policy_hists_path,
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=local_num_epochs,
                save_dir=args.model_savedir,
                num_classes=num_classes,
                ignore_index=ignore_index,
                model_name=args.model,
                dataset_name=args.dataset,
                metric="iou",
                start_epoch=local_start_epoch,
                attacker=train_attacker,
                eval_attacker=eval_attacker,
                adv_train=True,
                eval_interval=args.eval_interval,
                evaluate_robust=True,
                distributed=distributed,
                use_clean_weights=args.rl_apply_clean_weight,
                use_adv_weights=args.rl_apply_adv_weight,
                clean_weight_mode=args.rl_clean_weight_mode,
                adv_weight_mode=args.rl_adv_weight_mode,
                rl_action_epochs=args.rl_action_epochs,
                rl_learning_rate=args.rl_learning_rate,
                rl_hidden_dim=args.rl_hidden_dim,
                rl_entropy_coefficient=args.rl_entropy_coefficient,
                rl_reward_scale=args.rl_reward_scale,
                rl_uniform_mix=args.rl_uniform_mix,
                rl_min_weight=args.rl_min_weight,
                rl_max_weight=args.rl_max_weight,
                rl_checkpoint_path=args.rl_checkpoint_path
            )
        else:
            raise ValueError(f"Invalid training type: {args.training_type}")
    finally:
        dist_print(f"Training script finished. Results and checkpoints are saved at {args.model_savedir}")
        destroy_process_group(distributed=distributed)
        
if __name__ == "__main__":
    main()
    