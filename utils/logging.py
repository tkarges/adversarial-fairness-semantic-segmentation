import csv
import os
import torch
from cityscapes import CityscapesDataset

def tensor_to_float_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    
    return [
        entry.detach().cpu().item() if isinstance(entry, torch.Tensor) else float(entry)
        for entry in x
    ]
            
class Logger:
    def __init__(self, path):
        self.path = path
        file_exists = os.path.exists(path)
        self.f = open(path, 'a', newline='', buffering=1)
        self.writer = csv.writer(self.f)
        
        if not file_exists:
            self.writer.writerow(self.construct_header())
            self._sync()
    
    def construct_header(self):
        raise NotImplementedError('construct_header() missing in logger class')
    
    def log(self, *args, **kwargs):
        raise NotImplementedError('log() must be imlmeneted')
    
    def write_row(self, row):
        self.writer.writerow(row)
        self._sync()
        
    def _sync(self):
        self.f.flush()
        os.fsync(self.f.fileno())
    
    def close(self):
        if not self.f.closed:
            self.f.close()

class MetricsLogger(Logger):
    def __init__(self, path, num_classes, metric_name='iou', log_adv=False, log_weights=False):
        self.num_classes = num_classes
        self.metric_name = metric_name
        self.log_adv = log_adv
        self.log_weights = log_weights
        super().__init__(path)
            
    def construct_header(self):
        class_scores_header = [f"{self.metric_name}_class_{i}" for i in range(self.num_classes)]
        header = ['epoch', 'avg_loss', f'mean_{self.metric_name}', f'robust_mean_{self.metric_name}'] + class_scores_header
        
        if self.log_adv:
            class_scores_robust_header = [f'robust_{self.metric_name}_class_{i}' for i in range(self.num_classes)]
            header += class_scores_robust_header
        
        if self.log_weights:
            class_weights_header = [f'weight_class_{i}' for i in range(self.num_classes)]
            header += class_weights_header
            
        return header
    
    def log(self, epoch_idx, avg_loss, mean_score, class_scores, mean_score_robust=None, class_scores_robust=None, class_weights=None):
        row = [epoch_idx, avg_loss, mean_score]
        if self.log_adv and mean_score_robust is not None:
            row += [mean_score_robust]
        
        row += tensor_to_float_list(class_scores)
        
        if self.log_adv and class_scores_robust is not None:
            row += tensor_to_float_list(class_scores_robust)
            
        if self.log_weights and class_weights is not None:
            row += tensor_to_float_list(class_weights)

        self.write_row(row)
        
class EvalLogger(Logger):
    def __init__(self, path, num_classes=CityscapesDataset.num_classes):
        self.num_classes = num_classes
        super().__init__(path)
            
    def construct_header(self):
        header_tp = [f'TP_{i}' for i in range(self.num_classes)]
        header_fp = [f'FP_{i}' for i in range(self.num_classes)]
        header_fn = [f'FN_{i}' for i in range(self.num_classes)]
        return header_tp + header_fp + header_fn
    
    def log(self, tp, fp, fn):
        row = (
            tensor_to_float_list(tp)
            + tensor_to_float_list(fp)
            + tensor_to_float_list(fn)
        )
        self.write_row(row)
        
class CleanLogger(Logger):
    def __init__(self, path, metric, num_classes=CityscapesDataset.num_classes):
        self.num_classes = num_classes
        self.metric = metric
        super().__init__(path)
        
    def construct_header(self):
        class_scores_header = [f'{self.metric}_class_{i}' for i in range(self.num_classes)]
        return (
            ['epoch', 'avg_loss', f'mean_{self.metric}']
            + class_scores_header
        )
        
    def log(self, epoch_idx, avg_loss, mean_score, class_scores):
        row = (
            [epoch_idx, avg_loss, mean_score]
            + tensor_to_float_list(class_scores)
        )
        self.write_row(row)
        
class RLDecisionLogger:
    def __init__(self, path, num_classes, action_dim=4):
        self.path = path
        self.num_classes = int(num_classes)
        self.action_dim = int(action_dim)

        os.makedirs(
            os.path.dirname(path) or ".",
            exist_ok=True,
        )

        file_exists = os.path.exists(path)
        self.file = open(
            path,
            mode="a",
            newline="",
            buffering=1,
        )

        self.fieldnames = self._build_fieldnames()
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=self.fieldnames,
        )

        if not file_exists or os.path.getsize(path) == 0:
            self.writer.writeheader()

    def _build_fieldnames(self):
        fields = [
            "decision_step",
            "epoch",
            "reward",
            "scaled_reward",
            "baseline_before",
            "advantage",
            "policy_loss",
            "grad_norm",
            "log_prob",
            "entropy",
            "weight_min",
            "weight_max",
            "weight_mean",
            "weight_std",
            "clean_mean",
            "robust_mean",
            "robust_min",
            "robust_std",
        ]

        for idx in range(self.action_dim):
            fields.extend([
                f"policy_mean_{idx}",
                f"policy_std_{idx}",
                f"policy_log_std_{idx}",
                f"sampled_action_{idx}",
                f"coefficient_{idx}",
            ])

        for class_idx in range(self.num_classes):
            fields.extend([
                f"clean_score_{class_idx}",
                f"robust_score_{class_idx}",
                f"vulnerability_{class_idx}",
                f"candidate_weight_{class_idx}",
                f"active_weight_{class_idx}",
            ])

        return fields

    def log(
        self,
        *,
        decision_step,
        epoch,
        policy_diagnostics,
        weight_diagnostics,
        candidate_weights,
        active_weights,
        clean_scores,
        robust_scores,
        reward=None,
        scaled_reward=None,
        baseline_before=None,
        advantage=None,
        policy_loss=None,
        grad_norm=None,
    ):
        policy_mean = (
            policy_diagnostics["mean"]
            .detach()
            .float()
            .cpu()
        )
        policy_std = (
            policy_diagnostics["std"]
            .detach()
            .float()
            .cpu()
        )
        policy_log_std = (
            policy_diagnostics["log_std"]
            .detach()
            .float()
            .cpu()
        )
        sampled_action = (
            policy_diagnostics["action"]
            .detach()
            .float()
            .cpu()
        )

        coefficients = (
            weight_diagnostics["coefficients"]
            .detach()
            .float()
            .cpu()
        )
        vulnerability = (
            weight_diagnostics["vulnerability"]
            .detach()
            .float()
            .cpu()
        )

        candidate_weights = (
            candidate_weights.detach().float().cpu()
        )
        active_weights = (
            active_weights.detach().float().cpu()
        )
        clean_scores = (
            clean_scores.detach().float().cpu()
        )
        robust_scores = (
            robust_scores.detach().float().cpu()
        )

        row = {
            "decision_step": int(decision_step),
            "epoch": int(epoch),
            "reward": reward,
            "scaled_reward": scaled_reward,
            "baseline_before": baseline_before,
            "advantage": advantage,
            "policy_loss": policy_loss,
            "grad_norm": grad_norm,
            "log_prob": policy_diagnostics["log_prob"],
            "entropy": policy_diagnostics["entropy"],
            "weight_min": float(active_weights.min().item()),
            "weight_max": float(active_weights.max().item()),
            "weight_mean": float(active_weights.mean().item()),
            "weight_std": float(
                active_weights.std(unbiased=False).item()
            ),
            "clean_mean": float(clean_scores.mean().item()),
            "robust_mean": float(robust_scores.mean().item()),
            "robust_min": float(robust_scores.min().item()),
            "robust_std": float(
                robust_scores.std(unbiased=False).item()
            ),
        }

        for idx in range(self.action_dim):
            row[f"policy_mean_{idx}"] = float(
                policy_mean[idx].item()
            )
            row[f"policy_std_{idx}"] = float(
                policy_std[idx].item()
            )
            row[f"policy_log_std_{idx}"] = float(
                policy_log_std[idx].item()
            )
            row[f"sampled_action_{idx}"] = float(
                sampled_action[idx].item()
            )
            row[f"coefficient_{idx}"] = float(
                coefficients[idx].item()
            )

        for class_idx in range(self.num_classes):
            row[f"clean_score_{class_idx}"] = float(
                clean_scores[class_idx].item()
            )
            row[f"robust_score_{class_idx}"] = float(
                robust_scores[class_idx].item()
            )
            row[f"vulnerability_{class_idx}"] = float(
                vulnerability[class_idx].item()
            )
            row[f"candidate_weight_{class_idx}"] = float(
                candidate_weights[class_idx].item()
            )
            row[f"active_weight_{class_idx}"] = float(
                active_weights[class_idx].item()
            )

        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.close()