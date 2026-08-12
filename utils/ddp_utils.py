import torch
import torch.distributed as dist
import os

def setup_ddp():
    if not torch.cuda.is_available():
        return 0, torch.device("cpu"), False

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        return local_rank, torch.device(f"cuda:{local_rank}"), True

    return 0, torch.device("cuda:0"), False

def is_distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()

    # torchrun sets RANK globally
    if "RANK" in os.environ:
        return int(os.environ["RANK"])

    # fallback for non-distributed execution
    return 0

def is_main_process():
    return get_rank() == 0

def dist_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs, flush=True)
        
def reduce_sum(value, device):
    t = torch.tensor([value], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return t.item()

def is_initialized():
    return dist.is_initialized()

def destroy_process_group(distributed):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()