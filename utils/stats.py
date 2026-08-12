import torch
from utils.ddp_utils import dist_print

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic_light", "traffic_sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle",
]

CITYSCAPES_TAIL = [3, 4, 5, 6, 7, 12, 16, 17, 18]

def print_per_class_evolution(
    epoch,
    clean_scores,
    robust_scores=None,
    class_names=None,
    focus_classes=None,
    previous_clean=None,
    previous_robust=None,
):
    clean_scores = torch.as_tensor(clean_scores, dtype=torch.float32).detach().cpu()

    if robust_scores is not None:
        robust_scores = torch.as_tensor(robust_scores, dtype=torch.float32).detach().cpu()

    if focus_classes is None:
        focus_classes = list(range(clean_scores.numel()))

    lines = [f"\nPer-class scores at epoch {epoch}:"]

    for c in focus_classes:
        name = class_names[c] if class_names is not None else f"class_{c}"

        clean = clean_scores[c].item()
        msg = f"  {c:02d} {name:<14} clean={clean:.4f}"

        if previous_clean is not None:
            prev = previous_clean[c].item()
            msg += f" Δc={clean - prev:+.4f}"

        if robust_scores is not None:
            robust = robust_scores[c].item()
            msg += f" robust={robust:.4f}"

            if previous_robust is not None:
                prev_r = previous_robust[c].item()
                msg += f" Δr={robust - prev_r:+.4f}"

        lines.append(msg)

    dist_print("\n".join(lines))
    
def print_tail_summary(
    clean_scores,
    robust_scores=None,
    dataset_name="cityscapes",
    previous_clean=None,
    previous_robust=None,
):
    clean_scores = torch.as_tensor(clean_scores, dtype=torch.float32).detach().cpu()

    tail_classes = list(range(clean_scores.numel()))

    tail_amount = clean_scores.numel() // 2
    tail = torch.as_tensor(tail_classes, dtype=torch.long)[:tail_amount]

    clean_tail = clean_scores[tail].mean().item()
    msg = f"tail clean={clean_tail:.4f}"

    if previous_clean is not None:
        previous_clean = torch.as_tensor(previous_clean, dtype=torch.float32).detach().cpu()
        prev_clean_tail = previous_clean[tail].mean().item()
        msg += f", Δtail_clean={clean_tail - prev_clean_tail:+.4f}"

    if robust_scores is not None:
        robust_scores = torch.as_tensor(robust_scores, dtype=torch.float32).detach().cpu()
        robust_tail = robust_scores[tail].mean().item()
        msg += f", tail robust={robust_tail:.4f}"

        if previous_robust is not None:
            previous_robust = torch.as_tensor(previous_robust, dtype=torch.float32).detach().cpu()
            prev_robust_tail = previous_robust[tail].mean().item()
            msg += f", Δtail_robust={robust_tail - prev_robust_tail:+.4f}"

    dist_print(msg)