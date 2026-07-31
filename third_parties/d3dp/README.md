# D3DP -- VARPose Dense Information Injection

Fine-tunes the pretrained D3DP (17-joint sparse model) by injecting dense
144-joint 2D detections (161 joints in total) through cross-attention to improve
3D pose estimation on MPI-INF-3DHP.

## Directory Layout

Run all commands from this directory:

```bash
cd third_parties/d3dp
```

The expected files are:

```text
third_parties/d3dp/
├── data/
│   ├── data_train_3dhp_dense_annot3.npz
│   └── data_test_3dhp_dense_annot3.npz
├── checkpoints/
│   ├── 3dhp_finetune_work/
│   │   └── 3dhp_best_epoch.bin
│   └── d3dp_varpose_cross_attention_3dhp/
│       └── best_epoch.bin
├── scripts/
│   ├── train_3dhp_varpose_cross_attention.sh
│   └── test_3dhp_varpose_cross_attention.sh
├── 3dhp_test/
│   └── test_util/
│       └── mpii_test_predictions_ori_py.m
├── cal_complexity.py
└── main_3dhp.py
```

## Data

| File | Description |
| --- | --- |
| `data/data_train_3dhp_dense_annot3.npz` | MPI-INF-3DHP training annotations with dense 2D detections |
| `data/data_test_3dhp_dense_annot3.npz` | MPI-INF-3DHP test annotations with dense 2D detections |

## Checkpoints

| File | Description |
| --- | --- |
| `checkpoints/3dhp_finetune_work/3dhp_best_epoch.bin` | Pretrained sparse model (fine-tuning starting point) |
| `checkpoints/d3dp_varpose_cross_attention_3dhp/best_epoch.bin` | Fine-tuned dense cross-attention model (for evaluation) |

## Fine-tuning

Start from the pretrained D3DP sparse checkpoint and fine-tune with 161-joint 2D input:

```shell
./scripts/train_3dhp_varpose_cross_attention.sh
```

### Evaluation

```shell
./scripts/test_3dhp_varpose_cross_attention.sh
```

To produce the final MPJPE, PCK, and AUC report, run `3dhp_test/test_util/mpii_test_predictions_ori_py.m` in MATLAB. Update the load path on line 30:

```matlab
% original: load(['..\..\checkpoint\inference_data_' aggregation_mode '.mat'])
load('../../checkpoints/3dhp_finetune_work/H1_K1/inference_data_J_Agg.mat')
```

Output: `3dhp_test/test_util/mpii_3dhp_evaluation_sequencewise_ori_J_Agg_t1.csv` — 6 sequences (TS1–TS6) × 17 joints + Average column.

## Complexity

Use `cal_complexity.py` to compare the sparse 17-joint model with the dense
17+144-joint cross-attention model:

```shell
python cal_complexity.py --no-max-bs
```

The script reports MACs per frame, parameter count, throughput, latency, and
peak GPU memory. Omit `--no-max-bs` to additionally search for the maximum batch
size and benchmark throughput at that batch size:

```shell
python cal_complexity.py
```