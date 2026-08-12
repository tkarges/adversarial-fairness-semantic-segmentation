import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np


class WeightPolicy(nn.Module):
    '''
    Small neural network with one hidden layer that represents the policy.
    '''
    def __init__(self, state_dim, hidden_dim=32, initial_log_std=-2.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4),
        )
        
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        
        # Standard deviations are learned as separate global parameters
        self.log_std = nn.Parameter(torch.full((4,), float(initial_log_std)))

    def sample(self, state):
        # The policy outputs the mean of a Gaussian from a given state
        mean = self.net(state)
        log_std_clamped = self.log_std.clamp(-4.0, -0.25)
        std = log_std_clamped.exp()
        distribution = torch.distributions.Normal(mean, std)

        # An action is sampled from the distribution defined by the mean and log std parameters
        action = distribution.sample()
        log_prob = distribution.log_prob(action).sum()
        entropy = distribution.entropy().sum()
        
        diagnostics = {
            "mean": mean.detach(),
            "std": std.detach(),
            "log_std": log_std_clamped.detach(),
            "action": action.detach(),
            "log_prob": float(log_prob.detach().item()),
            "entropy": float(entropy.detach().item()),
        }
        
        return action, log_prob, entropy, diagnostics


class RewardBaseline:
    '''
    Baseline for reducing variance implemented as exponential moving average of rewards.
    '''
    def __init__(self, momentum=0.9):
        self.momentum = float(momentum)
        self.value = None

    def get(self):
        # Initialized to be 0
        return 0.0 if self.value is None else self.value

    def update(self, reward):
        reward = float(reward)
        if self.value is None:
            self.value = reward
        else:
            # Updates baseline
            self.value = self.momentum * self.value + (1.0 - self.momentum) * reward

    def state_dict(self):
        return {
            "momentum": self.momentum,
            "value": self.value,
        }

    def load_state_dict(self, state):
        self.momentum = float(state.get("momentum", 0.9))
        self.value = state.get("value")


def _standardize(values):
    '''
    Standardizes values to have 0 mean and unit variance
    '''
    values = values.float()
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)


def _safe_scores(values):
    '''
    Helper function that handles infinite scores and None values
    '''
    return torch.nan_to_num(
        values.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _lower_tail_mean(scores, fraction=0.5):
    '''
    Mean score over the lower tail of classes. In this case, the lower tail is 50% of all classes.
    '''
    scores = _safe_scores(scores)
    sorted_scores = torch.sort(scores).values
    count = max(1, int(round(sorted_scores.numel() * fraction)))
    return sorted_scores[:count].mean()


def build_policy_state(
    clean_scores,
    robust_scores,
    previous_robust_scores,
    epoch,
    end_epoch
):
    '''
    Builds a state from model validation feedback.
    
    Parameters:
    clean_scores: clean IoU scores of the current state of the segmentation model
    robust_scores: robust IoU scores of the current state of the segmentation model
    previous_robust_scores: robust IoU scores from the previous state of the model, before applying the current action
    epoch: current training epoch
    end_epoch: epoch in which training ends
    '''
    clean_scores = _safe_scores(clean_scores)
    robust_scores = _safe_scores(robust_scores)
    previous_robust_scores = _safe_scores(previous_robust_scores)

    robust_change = robust_scores - previous_robust_scores
    robustness_gap = (clean_scores - robust_scores).clamp_min(0.0)

    # Per class scores used in the state
    per_class = torch.stack(
        [
            _standardize(clean_scores),
            _standardize(robust_scores),
            _standardize(robustness_gap),
            _standardize(robust_change),
        ],
        dim=1,
    ).flatten()

    progress = float(epoch) / max(float(end_epoch - 1), 1.0)
    progress = min(max(progress, 0.0), 1.0)

    # Global statistics used in the state
    global_state = torch.tensor(
        [
            clean_scores.mean().item(),
            robust_scores.mean().item(),
            robust_scores.min().item(),
            robust_scores.std(unbiased=False).item(),
            progress
        ],
        dtype=torch.float32,
        device=clean_scores.device
    )

    # Final state of size 4C+5, where C = number of classes
    return torch.cat([per_class, global_state])


def action_to_class_weights(
    action,
    clean_scores,
    robust_scores,
    class_frequencies,
    fixed_target_classes,
    uniform_mix=0.30,
    min_weight=0.7,
    max_weight=1.6
):
    '''
    Computes class-wise loss weights from an action
    '''
    if fixed_target_classes is None:
        raise ValueError("fixed_target_classes must be initialized before generating RL weights.")

    if not 0.0 <= uniform_mix <= 1.0:
        raise ValueError(f"uniform_mix must be in [0, 1], got {uniform_mix}.")
    if not 0.0 < min_weight <= 1.0 <= max_weight:
        raise ValueError(
            "Expected 0 < min_weight <= 1 <= max_weight, got "
            f"[{min_weight}, {max_weight}]."
        )

    clean_scores = _safe_scores(clean_scores)
    robust_scores = _safe_scores(robust_scores)
    class_frequencies = class_frequencies.float().clamp_min(1e-12)

    fixed_target_classes = fixed_target_classes.to(
        device=robust_scores.device,
        dtype=torch.long
    )

    weakness = 1.0 - robust_scores
    robustness_gap = (clean_scores - robust_scores).clamp_min(0.0)

    frequency_probability = class_frequencies / class_frequencies.sum().clamp_min(1e-12)
    rarity = -torch.log(frequency_probability.clamp_min(1e-8))

    target_membership = torch.zeros_like(robust_scores)
    target_membership[fixed_target_classes] = 1.0

    signals = torch.stack(
        [
            _standardize(weakness),
            _standardize(robustness_gap),
            _standardize(rarity),
            _standardize(target_membership)
        ],
        dim=1
    )

    coefficients = F.softmax(action, dim=0)

    # Vulnerability score
    vulnerability = (signals * coefficients.unsqueeze(0)).sum(dim=1)
    vulnerability = _standardize(vulnerability)

    # Weights computed from vulnerability score
    raw_weights = torch.exp((0.6 * vulnerability).clamp(-4.0, 4.0))
    raw_weights = raw_weights / raw_weights.mean().clamp_min(1e-8)

    # Uniform weights are added for stability
    weights = uniform_mix * torch.ones_like(raw_weights) + (1.0 - uniform_mix) * raw_weights

    # Weights are normalized and clamped to be within the allowed range
    weights = weights.clamp(min=min_weight, max=max_weight)
    weights = weights / weights.mean().clamp_min(1e-8)
    weights = weights.clamp(min=min_weight, max=max_weight)
    weights = weights / weights.mean().clamp_min(1e-8)

    if not torch.isfinite(weights).all():
        raise RuntimeError(f"Non-finite RL class weights: {weights}")
    if (weights <= 0).any():
        raise RuntimeError(f"Non-positive RL class weights: {weights}")

    diagnostics = {
        "coefficients": coefficients.detach(),
        "vulnerability": vulnerability.detach(),
        "target_membership": target_membership.detach(),
        "weights": weights.detach(),
        "weight_min": weights.min().detach(),
        "weight_max": weights.max().detach(),
        "weight_std": weights.std(unbiased=False).detach()
    }

    return weights, diagnostics


def calculate_policy_reward(
    old_clean,
    old_robust,
    new_clean,
    new_robust,
    fixed_target_classes,
    dynamic_bottom_k=5,
    robust_drop_tolerance=0.005,
    clean_drop_tolerance=0.015,
    class_drop_tolerance=0.02,
    target_drop_tolerance=0.015
):
    '''
    Calculates the reward for an action
    '''
    old_clean = _safe_scores(old_clean)
    old_robust = _safe_scores(old_robust)
    new_clean = _safe_scores(new_clean)
    new_robust = _safe_scores(new_robust)

    fixed_target_classes = fixed_target_classes.to(
        device=new_robust.device,
        dtype=torch.long
    )

    old_fixed_bottom = old_robust[fixed_target_classes].mean()

    new_fixed_bottom = new_robust[fixed_target_classes].mean()

    delta_fixed_bottom = new_fixed_bottom - old_fixed_bottom

    old_lower_half = _lower_tail_mean(old_robust, fraction=0.5)

    new_lower_half = _lower_tail_mean(new_robust, fraction=0.5)

    delta_lower_half = new_lower_half - old_lower_half

    delta_robust_mean = new_robust.mean() - old_robust.mean()

    delta_clean_mean = new_clean.mean() - old_clean.mean()

    robust_collapse_penalty = torch.clamp(-delta_robust_mean - robust_drop_tolerance, min=0.0)

    clean_collapse_penalty = torch.clamp(-delta_clean_mean - clean_drop_tolerance, min=0.0)

    per_class_robust_drop = (old_robust - new_robust - class_drop_tolerance).clamp_min(0.0)

    class_collapse_penalty = per_class_robust_drop.mean()

    target_robust_drop = (
        old_robust[fixed_target_classes]
        - new_robust[fixed_target_classes]
        - target_drop_tolerance
    ).clamp_min(0.0)

    target_collapse_penalty = target_robust_drop.mean()

    reward = (
        1.50 * delta_fixed_bottom
        + 0.75 * delta_lower_half
        + 1.00 * delta_robust_mean
        + 0.25 * delta_clean_mean
        - 2.00 * robust_collapse_penalty
        - 1.00 * clean_collapse_penalty
        - 2.00 * class_collapse_penalty
        - 1.50 * target_collapse_penalty
    )

    k = min(int(dynamic_bottom_k), old_robust.numel())

    old_dynamic_indices = torch.topk(old_robust, k=k, largest=False).indices

    delta_dynamic_bottom = new_robust[old_dynamic_indices].mean() - old_robust[old_dynamic_indices].mean()

    diagnostics = {
        "reward": float(reward.item()),
        "old_fixed_bottom": float(old_fixed_bottom.item()),
        "new_fixed_bottom": float(new_fixed_bottom.item()),
        "delta_fixed_bottom": float(delta_fixed_bottom.item()),
        "old_lower_half": float(old_lower_half.item()),
        "new_lower_half": float(new_lower_half.item()),
        "delta_lower_half": float(delta_lower_half.item()),
        "delta_dynamic_bottom": float(delta_dynamic_bottom.item()),
        "delta_robust_mean": float(delta_robust_mean.item()),
        "delta_clean_mean": float(delta_clean_mean.item()),
        "robust_collapse_penalty": float(robust_collapse_penalty.item()),
        "clean_collapse_penalty": float(clean_collapse_penalty.item()),
        "class_collapse_penalty": float(class_collapse_penalty.item()),
        "target_collapse_penalty": float(target_collapse_penalty.item())
    }

    return float(reward.item()), diagnostics


@torch.no_grad()
def collect_class_frequencies(dataloader, num_classes, ignore_index, device):
    counts = torch.zeros(num_classes, dtype=torch.float64, device=device)

    for _, target in dataloader:
        target = target.to(device, non_blocking=device.type == "cuda")
        valid = ((target != ignore_index) & (target >= 0) & (target < num_classes))
        counts += torch.bincount(target[valid], minlength=num_classes).double()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)

    return counts.float().clamp_min(1.0)

def load_training_class_frequencies(histogram_path, split_idx_path, device):
    hists = np.load(histogram_path)
    split = torch.load(split_idx_path, map_location="cpu", weights_only=True)
    train_idx = np.asarray(split["train_main_idx"], dtype=np.int64)
    counts = hists[train_idx].sum(axis=0).astype(np.float64)
    frequencies = counts / np.clip(counts.sum(), 1.0, None)
    return torch.as_tensor(frequencies, dtype=torch.float32, device=device)

def broadcast_class_weights(weights, num_classes, device):
    if not (dist.is_available() and dist.is_initialized()):
        if weights is None:
            raise ValueError("Weights cannot be None outside DDP.")
        return weights.to(device=device, dtype=torch.float32)

    if dist.get_rank() == 0:
        if weights is None:
            raise ValueError("Rank 0 must provide class weights.")
        result = weights.to(device=device, dtype=torch.float32)
    else:
        result = torch.empty(
            num_classes,
            device=device,
            dtype=torch.float32,
        )

    dist.broadcast(result, src=0)
    return result


def save_rl_checkpoint(
    path,
    policy,
    optimizer,
    baseline,
    weights,
    previous_robust,
    rl_fixed_bottom_classes
):
    torch.save(
        {
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "baseline": baseline.state_dict(),
            "weights": weights.detach().cpu(),
            "previous_robust": (
                None
                if previous_robust is None
                else previous_robust.detach().cpu()
            ),
            "fixed_bottom_classes": (
                None
                if rl_fixed_bottom_classes is None
                else rl_fixed_bottom_classes.detach().cpu()
            )
        },
        path
    )


def load_rl_checkpoint(
    path,
    policy,
    optimizer,
    baseline,
    device
):
    state = torch.load(path, map_location=device)

    policy.load_state_dict(state["policy"])
    optimizer.load_state_dict(state["optimizer"])
    baseline.load_state_dict(state["baseline"])

    weights = state["weights"].to(device)

    previous_robust = state.get("previous_robust")
    if previous_robust is not None:
        previous_robust = previous_robust.to(device)

    fixed_bottom_classes = state.get("fixed_bottom_classes")
    if fixed_bottom_classes is not None:
        fixed_bottom_classes = fixed_bottom_classes.to(
            device=device,
            dtype=torch.long
        )

    return weights, previous_robust, fixed_bottom_classes


def select_fixed_bottom_classes(
    baseline_robust_scores,
    k=5,
):
    scores = baseline_robust_scores.detach().float()
    finite = torch.isfinite(scores)

    if finite.sum() < k:
        raise ValueError(
            f"Only {finite.sum().item()} finite class scores are available, but k={k}."
        )

    selection_scores = scores.clone()
    selection_scores[~finite] = float("inf")

    return torch.topk(
        selection_scores,
        k=k,
        largest=False
    ).indices