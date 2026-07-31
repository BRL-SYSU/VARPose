#!/bin/bash
# Evaluate the cross-attention (FusionPose) model on H36M with detected 2D ("sh").
# Uses the pretrained checkpoint at checkpoint/varpose_injection/best_model.pth.
#
# Prerequisites:
#   checkpoint/varpose_injection/best_model.pth
#   data/h36m/h36m_test_dense.pkl
#   data/h36m/h36m_sh_dt_ft_dense.pkl
#
# Expected results (--sample 640, 200 hypotheses):
#   Protocol 1 (MPJPE):     ~35.0 mm
#   Protocol 2 (PA-MPJPE):  ~29.9 mm
# Single-GPU quick check: set GPUS=1 and HYPO=1 below.
set -e
cd "$(dirname "$0")/.."

CKPT=checkpoint/varpose_injection
GPUS=4
HYPO=200

# dataset paths are filenames relative to data/h36m/ (the loader joins them
# with the dataset root internally).
COMMON=(--config configs/subvp/h36m_ncsnpp_deep_continuous.py
    --ckpt-dir "${CKPT}"
    --best
    --dataset h36m
    --dataset-path h36m_test_dense.pkl
    --detector-dataset-path h36m_sh_dt_ft_dense.pkl
    --gpus "${GPUS}"
    --hypo "${HYPO}"
    --sample 640)

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.eval_fc_mp_adv_3d \
    "${COMMON[@]}" est \
    2>&1 | tee output/test_h36m_sh_cross_attention_p1.log

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.eval_fc_mp_adv_3d \
    "${COMMON[@]}" --proto2 est \
    2>&1 | tee output/test_h36m_sh_cross_attention_p2.log
