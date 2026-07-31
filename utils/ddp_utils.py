"""
DDP utility functions module
Provides various utility functions required for distributed data parallel training
And supports compatibility between single-GPU and multi-GPU setups
"""
import os
import torch
import torch.distributed as dist
from datetime import timedelta

_DIST_INITIALIZED = False

def setup_ddp(backend='nccl', timeout_minutes=30):
    """
    Initialize the DDP environment.
    If torchrun/slurm environment variables are detected, initialize the distributed process group.
    Otherwise, treat it as single-GPU training.
    """
    global _DIST_INITIALIZED

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        try:
            # Get process info
            local_rank = int(os.environ["LOCAL_RANK"])
            global_rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])

            # Initialize distributed process group
            dist.init_process_group(
                backend=backend, 
                init_method='env://',
                timeout=timedelta(minutes=timeout_minutes),
                device_id=torch.device(f'cuda:{local_rank}')
            )
            
            # Set GPU device for current process
            torch.cuda.set_device(local_rank)
            
            _DIST_INITIALIZED = True
            print(f"DDP initialized: rank={global_rank}/{world_size}, local_rank={local_rank} on device cuda:{local_rank}")

        except Exception as e:
            print(f"Error initializing DDP: {e}")
            raise
    else:
        # Single GPU mode
        local_rank = 0
        global_rank = 0
        world_size = 1
        _DIST_INITIALIZED = False
        print("Running in single-GPU mode. DDP is not initialized.")
        
    return local_rank, global_rank, world_size


def cleanup_ddp():
    """Clean up DDP environment"""
    global _DIST_INITIALIZED
    if _DIST_INITIALIZED and dist.is_initialized():
        dist.destroy_process_group()
        _DIST_INITIALIZED = False
        print("DDP environment cleaned up.")


def is_dist_initialized():
    """Check whether the distributed environment is initialized"""
    return _DIST_INITIALIZED and dist.is_available() and dist.is_initialized()


def is_main_process():
    """Determine whether this is the main process"""
    return not is_dist_initialized() or dist.get_rank() == 0


def master_only(func):
    """Decorator: execute the function only on the main process"""
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)
    return wrapper


def all_reduce(tensor, op='sum'):
    """
    Perform a reduce operation on the tensor across all GPUs
    
    Args:
        tensor: The tensor to reduce
        op: Reduce operation type, supports 'sum', 'mean', 'max', 'min'
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
    """Get the total number of processes"""
    return dist.get_world_size() if is_dist_initialized() else 1


def get_rank():
    """Get the global rank of the current process"""
    return dist.get_rank() if is_dist_initialized() else 0


def get_local_rank():
    """Get the local rank of the current process"""
    return int(os.environ.get("LOCAL_RANK", 0))


def barrier():
    """Synchronize all processes"""
    if is_dist_initialized():
        dist.barrier()


def broadcast(tensor, src=0):
    """
    Broadcast a tensor from one process to all processes
    
    Args:
        tensor: The tensor to broadcast
        src: Source process rank
    """
    if is_dist_initialized():
        dist.broadcast(tensor, src=src)
    return tensor

def gather_distributed_results(local_results: list) -> list:
    """
    Gather results from all processes in distributed training.
    
    Args:
        local_results: Results from the current process
        
    Returns:
        List containing results from all processes (only on main process),
        or empty list on non-main processes
    """
    if not is_dist_initialized():
        return local_results
    
    # Gather results from all processes
    gathered_results = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_results, local_results)
    if not is_main_process():
        return []
    combined_results = []
    for proc_results in gathered_results:
        if proc_results is not None:
            combined_results.extend(proc_results)
    return combined_results

def recursive_to_device(obj, device):
    """
    Recursively move all tensors in an object to the specified device
    
    Args:
        obj: Any object, may contain tensors, dicts, lists, tuples, etc.
        device: Target device (e.g. 'cuda:0', 'cpu', torch.device, etc.)
        
    Returns:
        An object with the same structure as the input, but with all tensors moved to the specified device
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: recursive_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(recursive_to_device(v, device) for v in obj)
    return obj