<div align="center">

# MixSTE (HPE / HMR) — VARPose Dense Information Injection

VARPose injects dense 161-joint information into MixSTE for:

- Human3.6M 3D human pose estimation
- 3DPW cross-dataset evaluation
- Human3.6M-based HMR mesh recovery

</div>

<p align="center">
  <img src="assets/SittingDown_s1.gif" alt="Demo: SittingDown (S1)" width="360" />
</p>

---

## Directory layout

Run the commands below from `third_parties/mixste/` unless noted otherwise.

```text
third_parties/mixste/
├── assets/
├── checkpoints/
│   ├── mixste_concat_gt_f_81/
│   │   └── best_epoch.bin
│   └── mixste_gt_reproduced_f_81/
│       └── best_epoch.bin
├── data/
│   ├── data_2d_h36m_gt_contiguous_161j.npz
│   ├── data_3d_h36m_remapped.npz
│   ├── h36m_format_for_lifting_3dpw_occ_cpn_dense.npz
│   ├── h36m_format_for_lifting_3dpw_test_cpn_dense.npz
│   └── msst_data_h36m_vp3d_standard.pkl
├── output/
├── log/
├── run.py
├── prepare_for_mesh_eval_h36m.py
└── scripts/
```

## Data

| File | Description |
| --- | --- |
| `data/data_3d_h36m_remapped.npz` | Human3.6M 3D poses |
| `data/data_2d_h36m_gt_contiguous_161j.npz` | Dense 2D keypoints for Human3.6M training |
| `data/h36m_format_for_lifting_3dpw_test_cpn_dense.npz` | 3DPW test set in H36M lifting format |
| `data/h36m_format_for_lifting_3dpw_occ_cpn_dense.npz` | 3DPW-OCC test set in H36M lifting format |
| `data/msst_data_h36m_vp3d_standard.pkl` | 17-48-96 hierarchical data |

## Checkpoints

| File | Description |
| --- | --- |
| `checkpoints/mixste_concat_gt_f_81/best_epoch.bin` | Dense-injection MixSTE checkpoint |
| `checkpoints/mixste_gt_reproduced_f_81/best_epoch.bin` | Sparse 17-joint MixSTE baseline checkpoint |

## Quick start

### Human3.6M

**Training**

```shell
./scripts/train_gt_concat_varpose_161.sh
```

**Testing** (using pretrained checkpoint `./checkpoints/mixste_concat_gt_f_81`)

```shell
./scripts/test_gt_concat_varpose_161.sh
```

### 3DPW

Evaluate the H36M-trained checkpoints on 3DPW and 3DPW-OCC:

```shell
./scripts/test_3dpw_concat_varpose_161.sh
```

### HMR Mesh Recovery

Step 1: export predictions and convert them to the mesh-eval NPZ format:

```shell
./scripts/prepare_for_hmr_h36m.sh
```

This produces:

- `output/mixste_concat_gt_f_81.npz`
- `output/mixste_concat_gt_f_81_integrated.npz`

Step 2: run mesh evaluation with the integrated NPZ:

This command must be run from the VARPose repository root because it uses the
top-level `run.py` task dispatcher, not `third_parties/mixste/run.py`.

```shell
cd ../..
mkdir -p third_parties/mixste/log/mesh_result
CUDA_VISIBLE_DEVICES=7 python -u run.py \
  --task smpl_mesh_eval_task \
  --npz_path third_parties/mixste/output/mixste_concat_gt_f_81_integrated.npz \
  --h36m2smpl_checkpoint checkpoints/h36m2smpl_layer6/best_model.pth \
  --num_transformer_layers 6 \
  --num_gcn_layers 6 \
  --batch_size 512 \
  --save_prediction third_parties/mixste/checkpoints/mesh_eval_output/161_varpose_mesh_params.npz \
  2>&1 | tee third_parties/mixste/log/mesh_result/161_varpose.log
```

## Model complexity analysis

Use `cal_complexity.py` to benchmark compute and performance, comparing the sparse (17-joint) and dense-injection (161-joint) configurations:

```shell
# Full benchmark: includes max batch size search (binary search to avoid OOM)
python cal_complexity.py

# Skip max batch size search, measure single-batch metrics only
python cal_complexity.py --no-max-bs
```
---

<div align="center">
<i>This repository is a research reproduction of MixSTE.</i>
</div>
