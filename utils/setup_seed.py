import random
import numpy as np
import torch


def setup_seed(seed):
    """Set random seed
    
    Args:
        seed: The given seed value. If negative, use a random seed strategy
    """
    # Check if random seed strategy is needed
    if seed < 0:
        seed = random.randint(1, 1000000000)
        print(f"Using random seed strategy, generated seed: {seed}")
    else:
        print(f"Using fixed seed: {seed}")
    
    # Set all random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure CUDA reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return seed
