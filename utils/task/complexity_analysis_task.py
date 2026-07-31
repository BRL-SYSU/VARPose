import thop
import torch
import gc

from .base_task import *
from models.vqvae_hierarchical_skeleton import HierarchicalVQVAE
from models.var_skeleton import VARForSkeleton
from data.h36m_msst_dataset import H36M_MSST_Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

class VQVAEWrapper(torch.nn.Module):
    def __init__(self, vqvae):
        super().__init__()
        self.vqvae:HierarchicalVQVAE = vqvae
    def forward(self, x17, x48, x96):
        self.vqvae.quantizer.training_cache_reset()
        return self.vqvae([x17, x48, x96], hasLoss=False)

class VARTrainWrapper(torch.nn.Module):
        def __init__(self, var_m):
            super().__init__()
            self.var_m:VARForSkeleton = var_m
            self.stage_latencies_ms = None
        def forward(self, lr_inp):
            out = self.var_m.inference(lr_inp, enable_timing=True)
            self.stage_latencies_ms = out.get("stage_latencies_ms", None)
            return out
    

def make_model(model_desc)->torch.nn.modules:
    def build_adj_matrices():
        adj_tuples = H36M_MSST_Dataset.get_adj_tuples_symmetry_augmented()
        return [adj_tuples[gt] for gt in [17, 48, 96]]
    if model_desc in ["HierarchicalVQVAE", "VARForSkeleton"]:
        vqvae = HierarchicalVQVAE(
            vocab_size=4096, embedding_dim=128, feature_dim=2,
            beta=0.25, using_znorm=False, quant_resi=0.5,
            v_patch_nums=(48, 102, 192, 288, 432, 576),
            gt_patch_nums=(17, 48, 96),
            adj_matrices=build_adj_matrices(),
            mlp_mixer_blocks=4, dropout_rate=0.0,
        ).to(device)
        vqvae.eval()
        if model_desc == "VARForSkeleton":
            model = VARForSkeleton(
                vae_local=vqvae, depth=4, embed_dim=256, num_heads=16,
                mlp_ratio=4.0, drop_rate=0.0,
                patch_nums=(48, 102, 192, 288, 432, 576),
                inp_seq_len=17, inp_feature_dim=2, sos_method="linear",
            ).to(device)
            model.eval()
            return VARTrainWrapper(model)
        else:
            return VQVAEWrapper(vqvae)
    else:
        raise RuntimeError("--model must in [\"HierarchicalVQVAE", "VARForSkeleton\"]")

def make_input(model_desc, bs=1)->torch.Tensor:
    if model_desc == "HierarchicalVQVAE":
        input = (torch.randn(bs, 17, 2, device=device),
                   torch.randn(bs, 48, 2, device=device),
                   torch.randn(bs, 96, 2, device=device))
    elif model_desc == "VARForSkeleton":
        input = (torch.randn(bs, 17, 2, device=device),)
    return input

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
    model.cpu()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    model.eval()
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

    stage_latencies_ms = None
    if hasattr(model, 'stage_latencies_ms') and model.stage_latencies_ms is not None:
        stage_latencies_ms = model.stage_latencies_ms

    del dummy
    torch.cuda.empty_cache()
    return fps, avg_ms_per_batch, batch_size, peak_memory_str, stage_latencies_ms



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

    fps, avg_ms_per_batch, real_bs, peak_memory_str, stage_latencies_ms = measure_fps(model, 1, make_input_params, num_frames)
    print(f"batch_size:{real_bs} fps: {fps:.3f} Frame/s latency: {avg_ms_per_batch:.3f} ms/batch Peak Memory:{peak_memory_str}")
    if stage_latencies_ms is not None:
        print(f"Per-stage latencies (ms):")
        for i, latency in enumerate(stage_latencies_ms):
            print(f"  Stage {i}: {latency:.3f} ms")

    if w_max_bs:
        max_bs = find_max_batch_size(model, make_input_params)
        print(f"find max bs: {max_bs}")
        fps, avg_ms_per_batch, real_bs, peak_memory_str, stage_latencies_ms = measure_fps(model, max_bs, make_input_params, num_frames)
        print(f"batch_size:{real_bs} fps: {fps:.3f} Frame/s latency: {avg_ms_per_batch:.3f} ms/batch Peak Memory:{peak_memory_str}")
        if stage_latencies_ms is not None:
            print(f"Per-stage latencies (ms):")
            for i, latency in enumerate(stage_latencies_ms):
                print(f"  Stage {i}: {latency:.3f} ms")
    print("="*50)


class ComplexityAnalysisTask(BaseTask):
    def __init__(self, args):
        super().__init__(args)
    
    @staticmethod
    def add_parser_args(parser):
        g = parser.add_argument_group('Complexity Analysis Task')
        g.add_argument('--model', type=str, required=True, 
                       choices=["HierarchicalVQVAE", "VARForSkeleton"],
                       help='Model to analyze complexity')
        g.add_argument("--frames-per-batch", type=int, default=1, help="Frames per batch")
        g.add_argument("--no-max-bs", action="store_true", help="Skip max batch size search")
        return parser
    
    def run(self):
        print(f"model_desc: {self.args.model}")
        make_model_params = {"model_desc" : self.args.model}
        make_input_params = {"model_desc" : self.args.model}
        benchmark_model(make_model_params, make_input_params, self.args.frames_per_batch, not self.args.no_max_bs)