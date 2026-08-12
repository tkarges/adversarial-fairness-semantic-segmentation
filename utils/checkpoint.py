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
    strict=True,
    load_optimizer=True,
    load_scheduler=True,
):
    checkpoint = torch.load(save_dir, map_location=device)
    print(checkpoint.keys())
    print(checkpoint["epoch"])

    model_state = checkpoint["model"]

    model_is_wrapped = hasattr(model, "module")
    model_state_has_module = (
        len(model_state) > 0
        and next(iter(model_state)).startswith("module.")
    )

    if model_is_wrapped and not model_state_has_module:
        model_state = _add_module_prefix(model_state)

    if not model_is_wrapped and model_state_has_module:
        model_state = _strip_module_prefix(model_state)

    model.load_state_dict(model_state, strict=strict)

    start_epoch = checkpoint.get("epoch", -1) + 1

    if load_optimizer and optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    if load_scheduler and scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return start_epoch