# GFPose (HPE) -- VARPose Dense Information Injection

Fine-tunes the pretrained GFPose U3D model by injecting dense 144-joint 2D
detections (161 joints in total) through the FusionPose cross-attention module
to improve 3D human pose estimation on Human3.6M.

## Directory Layout

Run all commands from this directory:

```bash
cd third_parties/gfpose
```

The expected files are:

```text
third_parties/gfpose/
├── data/
│   └── h36m/
│       ├── h36m_train_dense.pkl
│       ├── h36m_test_dense.pkl
│       └── h36m_sh_dt_ft_dense.pkl
├── checkpoint/
│   ├── u3d/
│   │   └── best_model.pth
│   └── varpose_injection/
│       └── best_model.pth
├── output/
├── scripts/
│   ├── train_h36m_sh_cross_attention.sh
│   └── test_h36m_sh_cross_attention.sh
└── run/
    ├── train_fc_adv_3d.py
    └── eval_fc_mp_adv_3d.py
```

## Data

| File | Description |
| --- | --- |
| `data/h36m/h36m_train_dense.pkl` | Human3.6M train set with 3D labels and dense 2D poses |
| `data/h36m/h36m_test_dense.pkl` | Human3.6M test set with 3D labels and dense 2D poses |
| `data/h36m/h36m_sh_dt_ft_dense.pkl` | Detected dense 2D poses used as conditioning input for training and evaluation |

## Checkpoints

| File | Description |
| --- | --- |
| `checkpoint/u3d/best_model.pth` | Pretrained U3D sparse model (fine-tuning starting point) |
| `checkpoint/varpose_injection/best_model.pth` | Fine-tuned dense cross-attention model (for evaluation) |

## Fine-tuning

Fine-tunes the cross-attention module from the U3D base model with grouped
learning rates (FusionPose at full LR, backbone at 0.1×).

```shell
./scripts/train_h36m_sh_cross_attention.sh
```

## Evaluation

Evaluate the fine-tuned checkpoint with 200 hypotheses across 4 GPUs:

```shell
./scripts/test_h36m_sh_cross_attention.sh
```

The script evaluates both Human3.6M protocols and writes their logs to:

| Protocol | Metric | Log | Expected result |
| --- | --- | --- | --- |
| Protocol 1 | MPJPE | `output/test_h36m_sh_cross_attention_p1.log` | ~35.0 mm |
| Protocol 2 | PA-MPJPE | `output/test_h36m_sh_cross_attention_p2.log` | ~29.9 mm |
