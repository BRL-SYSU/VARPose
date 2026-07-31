# FinePOSE — VARPose Dense Information Injection

Fine-tunes the pretrained FinePOSE (17-joint sparse model) by injecting dense 144-joint 2D detections (161 joints in total) to improve 3D lifting accuracy.

## Directory Layout

Run all commands from this directory:

```bash
cd third_parties/finepose
```

The expected files are:

```text
third_parties/finepose/
├── data/
│   ├── data_2d_h36m_cpn_contiguous_161j.npz
│   └── data_3d_h36m.npz
├── checkpoints/
│   ├── cpn_sparse_best/
│   │   └── best_epoch_20_10.bin
│   └── cpn_dense/
│       └── best_epoch_32_40.79.bin
├── scripts/
│   ├── train_finetune_version.sh
│   └── test_finetune_version.sh
└── main.py
```

## Data

| File | Description |
| --- | --- |
| `data/data_2d_h36m_cpn_contiguous_161j.npz` | 2D keypoints (CPN detections, 161 joints) |
| `data/data_3d_h36m.npz` | 3D ground truth (Human3.6M, VideoPose3D format) 

## Checkpoints

| File | Description |
| --- | --- |
| `checkpoints/cpn_sparse_best/best_epoch_20_10.bin` | Pretrained sparse model (finetune starting point, auto-loaded by `main.py` during training) |
| `checkpoints/cpn_dense/best_epoch_32_40.79.bin` | Finetuned dense model (for evaluation) |

## Fine-tuning

Start from the original FinePOSE sparse checkpoint and fine-tune with 161-joint
2D input:

```shell
./scripts/train_finetune_version.sh
```

### Evaluation

```shell
./scripts/test_finetune_version.sh
```
