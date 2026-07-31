#!/bin/bash
# Fine-tune the cross-attention (FusionPose) model on H36M with detected 2D ("sh").
# Loads the pretrained U3D base model and fine-tunes with grouped learning rates:
# FusionPose params at full LR, backbone at 0.1x LR.
#
# Prerequisites:
#   checkpoint/u3d/best_model.pth
#   data/h36m/h36m_train_dense.pkl
#   data/h36m/h36m_test_dense.pkl
#   data/h36m/h36m_sh_dt_ft_dense.pkl
#
# Output: output/h36m_h36m/<timestamp>-h36m_sh_cross_attention/
set -e
cd "$(dirname "$0")/.."

# dataset paths are filenames relative to data/h36m/ (the loader joins them
# with the dataset root internally).
CUDA_VISIBLE_DEVICES=0 python -m run.train_fc_adv_3d \
    --config configs/subvp/h36m_ncsnpp_deep_continuous.py \
    --finetune-from checkpoint/u3d/best_model.pth \
    --backbone-lr-scale 0.1 \
    --train-dataset-path h36m_train_dense.pkl \
    --test-dataset-path h36m_test_dense.pkl \
    --detector-dataset-path h36m_sh_dt_ft_dense.pkl \
    --name h36m_sh_cross_attention \
    2>&1 | tee output/train_h36m_sh_cross_attention.log
