import os
import torch


def _strip_module_prefix(state_dict):
    return {
        k.replace("module.", "", 1) if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }


def _add_module_prefix(state_dict):
    return {
        k if k.startswith("module.") else f"module.{k}": v
        for k, v in state_dict.items()
    }


def save_checkpoint(
    save_name,
    save_dir,
    model,
    optimizer=None,
    epoch=0,
    scheduler=None,
    save_unwrapped=True,
):
    os.makedirs(save_dir, exist_ok=True)

    model_to_save = model.module if hasattr(model, "module") else model

    checkpoint = {
        "epoch": epoch,
        "model": model_to_save.state_dict() if save_unwrapped else model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    save_path = os.path.join(save_dir, f"{save_name}.pth")
    torch.save(checkpoint, save_path)

    return save_path


def load_checkpoint(
    save_dir,
    model,
    optimizer=None,
    scheduler=None,
    device="cpu",
    load_optimizer=True,
    load_scheduler=True,
):
    checkpoint = torch.load(
        save_dir,
        map_location=device,
        weights_only=True,
    )

    model_state = checkpoint.get("model", checkpoint["state_dict"])
    model_state = _strip_module_prefix(model_state)

    target_model = model.module if hasattr(model, "module") else model
    target_state = target_model.state_dict()

    if sum(k in target_state for k in model_state) < sum(
        f"model.{k}" in target_state for k in model_state
    ):
        model_state = {
            f"model.{k}": v
            for k, v in model_state.items()
        }

    for key in ("mean", "std"):
        if key in target_state and key not in model_state:
            model_state[key] = target_state[key]

    for key in list(model_state):
        if (
            key in target_state
            and model_state[key].shape != target_state[key].shape
        ):
            del model_state[key]

    target_model.load_state_dict(model_state, strict=False)

    if load_optimizer and optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if load_scheduler and scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint.get("epoch", -1) + 1