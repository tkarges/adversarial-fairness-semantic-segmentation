import argparse
import torch
from models.load_model import load_dense_model
from evaluate import evaluate
from attacks.segpgd import SegPGD
from datasets.metadata import get_dataset_info, get_label_mapping
from datasets.loading import build_dense_dataloaders
from utils.checkpoint import load_checkpoint
import os
from datasets.metadata import get_dataset_info
from datasets.loading import build_dense_dataloaders
from models.load_model import load_dense_model
from utils.checkpoint import load_checkpoint

def compute_precision(confusion_matrix, eps=1e-7):
    tp = torch.diag(confusion_matrix).float()
    fp = confusion_matrix.sum(dim=0).float() - tp
    
    pr_per_class = tp / (tp + fp + eps)
    valid = (tp + fp) > 0
    mpr = pr_per_class[valid].mean()
    
    return mpr, pr_per_class, valid

def compute_recall(confusion_matrix, eps=1e-7):
    tp = torch.diag(confusion_matrix).float()
    fn = confusion_matrix.sum(dim=1).float() - tp
    
    rc_per_class = tp / (tp + fn + eps)
    valid = (tp + fn) > 0
    mrc = rc_per_class[valid].mean()
    
    return mrc, rc_per_class, valid

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deeplabv3")
    parser.add_argument("--dataset", type=str, default="cityscapes")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--result-path", type=str, required=True)
    parser.add_argument("--result-name", type=str, required=True)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--evaluate-clean", action="store_true")
    parser.add_argument("--evaluate-robust", action="store_true")
    
    parser.add_argument("--attacker", type=str, default="segpgd")
    parser.add_argument("--attack-iterations", type=int, default=3)
    parser.add_argument("--attack-epsilon", type=float, default=8.0/255)
    parser.add_argument("--attack-alpha", type=float, default=2.0/255)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.result_path):
        raise ValueError(f"Result path does not exist")
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    num_classes, ignore_index, data_root, easy_classes, hard_classes = get_dataset_info(args.dataset)
    
    _, dataloader, _, _, _, _ = build_dense_dataloaders(
        dataset=args.dataset,
        data_root=data_root,
        distributed=False,
        eval_batch_size=args.eval_batch_size,
        prob_class_crop=0.0,
        use_rl=False
    )
    
    model = load_dense_model(
        model_name=args.model,
        num_classes=num_classes,
        device=device,
        distributed=False,
        load_existing=True
    )
    
    _ = load_checkpoint(
        save_dir=args.checkpoint_path,
        model=model,
        optimizer=None,
        scheduler=None, 
        device=device,
        load_optimizer=False,
        load_scheduler=False
    )
    
    if args.evaluate_clean:
        eval_results_clean = evaluate(
            model=model, 
            dataloader=dataloader, 
            device=device, 
            compute_pred_bias=True, 
            num_classes=num_classes, 
            ignore_index=ignore_index, 
            easy_classes=easy_classes, 
            hard_classes=hard_classes
        )
        
        pr, pr_per_class, _ = compute_precision(eval_results_clean['confusion_matrix'])
        rc, rc_per_class, _ = compute_recall(eval_results_clean["confusion_matrix"])
        
        eval_results_clean["pr"] = pr
        eval_results_clean["pr_per_class"] = pr_per_class
        eval_results_clean["rc"] = rc
        eval_results_clean["rc_per_class"] = rc_per_class
        print(eval_results_clean)
        save_path_clean = f"{args.result_path}/cleanres_{args.result_name}.pt"
        torch.save(eval_results_clean, save_path_clean)
    
    if args.evaluate_robust:
        save_path_adv = f"{args.result_path}/segpgd3res_{args.result_name}.pt"
        
        attacker = SegPGD(
            iterations=args.attack_iterations,
            epsilon=args.attack_epsilon,
            alpha=args.attack_alpha,
            num_classes=num_classes,
            ignore_index=ignore_index
        )
        
        eval_results_adv = evaluate(
            model=model, 
            dataloader=dataloader, 
            attacker=attacker,
            device=device, 
            compute_pred_bias=True, 
            num_classes=num_classes, 
            ignore_index=ignore_index, 
            easy_classes=easy_classes, 
            hard_classes=hard_classes
        )
            
        pr, pr_per_class, _ = compute_precision(eval_results_adv['confusion_matrix'])
        rc, rc_per_class, _ = compute_recall(eval_results_adv["confusion_matrix"])
            
        eval_results_adv["pr"] = pr
        eval_results_adv["pr_per_class"] = pr_per_class
        eval_results_adv["rc"] = rc
        eval_results_adv["rc_per_class"] = rc_per_class
        print(eval_results_adv)
        torch.save(eval_results_adv, save_path_adv)
    
if __name__ == "__main__":
    main()    