import torch
import gc
import thop
import argparse

from common.diffusionpose_3dhp import D3DP

device = "cuda" if torch.cuda.is_available() else "cpu"

class Args:
    number_of_frames = 243
    timestep = 1000
    scale = 1.0
    cs = 512
    dep = 8
    test_time_augmentation = False


F = 243
SPARSE = 17
DENSE = 144
K = 1
H = 1

JOINTS_L = [5, 6, 7, 11, 12, 13]
JOINTS_R = [2, 3, 4, 8, 9, 10]

CONFIGS = [
    ("Sparse (17j)",              0,   "cross_attention"),
    ("Dense-CrossAttn (17+144j)", 144, "cross_attention"),
]


def make_model(dense_j, strategy):
    args = Args()
    args.only_17 = (dense_j == 0)
    args.dense_fuse_strategy = strategy
    m = D3DP(args, JOINTS_L, JOINTS_R, is_train=False,
             num_proposals=H,
             sampling_timesteps=K,
             num_joints=SPARSE, num_joints_dense=dense_j,
             dense_fuse_strategy=strategy)
    m = m.to(device)
    m.device = torch.device(device)
    m.eval()
    return m

def make_input(dense_j, bs=1)->tuple[torch.Tensor]:
    n2d = SPARSE + dense_j
    x2d = torch.randn(bs, F, n2d, 2, device=device, dtype=torch.float)
    x3d = torch.randn(bs, F, SPARSE, 3, device=device, dtype=torch.float)
    # print(f"input_shape: {input.shape}")
    return (x2d, x3d)

def find_max_batch_size(model, make_input_params:dict):
    """Binary search for max batch size without OOM."""
    model.eval()
    # start from a reasonable lower bound
    lo, hi = 1, 2048
    max_bs = 0

    # first probe to find upper bound
    while hi > lo:
        mid = (lo + hi + 1) // 2
        try:
            torch.cuda.empty_cache()
            make_input_params["bs"]=mid
            dummy = make_input(**make_input_params)
            with torch.no_grad():
                _ = model(*dummy)
            del dummy
            torch.cuda.empty_cache()
            max_bs = mid
            lo = mid
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "OutOfMemoryError" in str(e):
                torch.cuda.empty_cache()
                gc.collect()
                hi = mid - 1
            else:
                raise RuntimeError("collect trash error")
            
        torch.cuda.empty_cache()
        gc.collect()
    return max_bs if max_bs > 0 else lo


def measure_fps(model, batch_size, make_input_params:dict, num_frames=1):
    """Measure FPS with cuda events for accurate timing."""
    warm_iters = 3
    test_iters = 10
    print(f"warm_iters: {warm_iters} test_iters: {test_iters}")
    torch.cuda.empty_cache()
    gc.collect()
    model.cpu()
    torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    make_input_params["bs"]=batch_size
    dummy = make_input(**make_input_params)

    # warmup
    with torch.no_grad():
        for _ in range(warm_iters):
            try:
                _ = model(*dummy)
            except:
                make_input_params["bs"] = make_input_params["bs"]//2
                batch_size = batch_size//2
                dummy = make_input(**make_input_params) 
                print(f"max batch size down to {make_input_params['bs']}")
    torch.cuda.synchronize()

    # measure
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        for _ in range(test_iters):
            _ = model(*dummy)
    end_event.record()
    torch.cuda.synchronize()
    peak_memory_bytes = torch.cuda.max_memory_allocated(device)
    peak_memory_str = thop.clever_format([peak_memory_bytes], "%.3f")

    elapsed_ms = start_event.elapsed_time(end_event)
    avg_ms_per_batch = elapsed_ms / test_iters
    fps = (batch_size * num_frames) / (avg_ms_per_batch / 1000.0)  # frames per second

    del dummy
    torch.cuda.empty_cache()
    return fps, avg_ms_per_batch, batch_size, peak_memory_str



def benchmark_model(make_model_params:dict, make_input_params:dict, num_frames=1, w_max_bs:bool=True):
    print("="*50)
    print(f"model config:\n{make_model_params}")
    print(f"input config:\n{make_input_params}")

    model = make_model(**make_model_params)
    make_input_params["bs"]=1
    dummy = make_input(**make_input_params)
    with torch.no_grad():
        macs, params = thop.profile(model, dummy, verbose=False)
    macs, params = thop.clever_format([macs/num_frames, params], "%.3f")
    print(f"Macs/frame: {macs} Params: {params}")

    fps, avg_ms_per_batch, real_bs, peak_memory_str = measure_fps(model, 1, make_input_params, num_frames)
    print(f"batch_size:{real_bs} fps: {fps:.3f} Frame/s latency: {avg_ms_per_batch:.3f} ms/batch Peak Memory:{peak_memory_str}")

    if w_max_bs:
        max_bs = find_max_batch_size(model, make_input_params)
        print(f"find max bs: {max_bs}")
        fps, avg_ms_per_batch, real_bs, peak_memory_str = measure_fps(model, max_bs, make_input_params, num_frames)
        print(f"batch_size:{real_bs} fps: {fps:.3f} Frame/s latency: {avg_ms_per_batch:.3f} ms/batch Peak Memory:{peak_memory_str}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-max-bs", action="store_true", help="Skip max batch size search")
    args = parser.parse_args()
    num_frames = Args.number_of_frames
    print(f"num_frames: {num_frames}")

    for config in CONFIGS:
        desc, dense_j, strategy = config
        print(desc)
        make_model_params = {"dense_j": dense_j, "strategy": strategy}
        make_input_params = {"dense_j": dense_j}
        benchmark_model(make_model_params, make_input_params, num_frames, not args.no_max_bs)

if __name__=="__main__":
    main()
