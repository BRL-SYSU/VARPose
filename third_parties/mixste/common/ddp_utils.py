"""
DDP utility module.
Provides utility functions required for distributed data parallel training
and supports both single-GPU and multi-GPU compatibility.
"""
import os
import torch
import torch.distributed as dist
from datetime import timedelta

_DIST_INITIALIZED = False

def setup_ddp(backend='nccl', timeout_minutes=30):
    """
    Initialize the DDP environment.
    If torchrun/slurm environment variables are detected, initialize the
    distributed process group. Otherwise, treat it as single-GPU training.
    """
    global _DIST_INITIALIZED

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        try:
            # Get process information
            local_rank = int(os.environ["LOCAL_RANK"])
            global_rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])

            # Start the distributed process group
            dist.init_process_group(
                backend=backend, 
                init_method='env://',
                timeout=timedelta(minutes=timeout_minutes)
            )
            
            # Set the GPU device used by the current process
            torch.cuda.set_device(local_rank)
            
            _DIST_INITIALIZED = True
            print(f"DDP initialized: rank={global_rank}/{world_size}, local_rank={local_rank} on device cuda:{local_rank}")

        except Exception as e:
            print(f"Error initializing DDP: {e}")
            raise
    else:
        # Single-GPU mode
        local_rank = 0
        global_rank = 0
        world_size = 1
        _DIST_INITIALIZED = False
        print("Running in single-GPU mode. DDP is not initialized.")
        
    return local_rank, global_rank, world_size


def cleanup_ddp():
    """Clean up the DDP environment."""
    global _DIST_INITIALIZED
    if _DIST_INITIALIZED and dist.is_initialized():
        dist.destroy_process_group()
        _DIST_INITIALIZED = False
        print("DDP environment cleaned up.")


def is_dist_initialized():
    """Check whether the distributed environment has been initialized."""
    return _DIST_INITIALIZED and dist.is_available() and dist.is_initialized()


def is_main_process():
    """Return whether this is the main process."""
    return not is_dist_initialized() or dist.get_rank() == 0


def master_only(func):
    """Decorator: run the wrapped function only on the main process."""
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)
    return wrapper


def all_reduce(tensor, op='sum'):
    """
    Reduce a tensor across all GPUs.
    
    Args:
        tensor: Tensor to reduce.
        op: Reduce operation type; supports 'sum', 'mean', 'max', and 'min'.
    """
    if not is_dist_initialized():
        return tensor

    if op == 'sum':
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    elif op == 'mean':
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= get_world_size()
    elif op == 'max':
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    elif op == 'min':
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    else:
        raise ValueError(f"Unsupported reduce operation: {op}")
    return tensor


def get_world_size():
    """Get the total number of processes."""
    return dist.get_world_size() if is_dist_initialized() else 1


def get_rank():
    """Get the global rank of the current process."""
    return dist.get_rank() if is_dist_initialized() else 0


def get_local_rank():
    """Get the local rank of the current process."""
    return int(os.environ.get("LOCAL_RANK", 0))


def barrier():
    """Synchronize all processes."""
    if is_dist_initialized():
        dist.barrier()


def broadcast(tensor, src=0):
    """
    Broadcast a tensor from one process to all processes.
    
    Args:
        tensor: Tensor to broadcast.
        src: Source process rank.
    """
    if is_dist_initialized():
        dist.broadcast(tensor, src=src)
    return tensor
