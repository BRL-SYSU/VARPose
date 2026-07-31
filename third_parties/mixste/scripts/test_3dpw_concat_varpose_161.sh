#! /bin/shell
# we select norm scale for the best pa-mpjpe with step 0.1.

# 3dpw Test Set with Injection of Dense Input
python run.py \
    --evaluate=best_epoch.bin \
    --checkpoint=./checkpoints/mixste_concat_gt_f_81 \
    --log=log/3dpw_test_varpose \
    --batch-size=81 \
    -f 81 \
    --keypoints=gt_contiguous_161j \
    -gpu=5 \
    --eval-3dpw \
    --data-path-3dpw=data/h36m_format_for_lifting_3dpw_test_cpn_dense.npz \
    --norm-scale=1.1

# 3dpw Test Set without Injection of Dense Input
python run.py \
    --evaluate=best_epoch.bin \
    --checkpoint=./checkpoints/mixste_gt_reproduced_f_81 \
    --log=log/3dpw_test_mixste \
    --batch-size=81 \
    -f 81 \
    --keypoints=gt_contiguous_161j \
    -gpu=5 \
    --eval-3dpw \
    --data-path-3dpw=data/h36m_format_for_lifting_3dpw_test_cpn_dense.npz \
    --only_17 \
    --norm-scale=1.1

# 3dpw OCC Set with Injection of Dense Input
python run.py \
    --evaluate=best_epoch.bin \
    --checkpoint=./checkpoints/mixste_concat_gt_f_81 \
    --log=log/3dpw_occ_varpose \
    --batch-size=81 \
    -f 81 \
    --keypoints=gt_contiguous_161j \
    -gpu=4 \
    --eval-3dpw \
    --data-path-3dpw=data/h36m_format_for_lifting_3dpw_occ_cpn_dense.npz \
    --norm-scale=1.2

# 3dpw OCC Set without Injection of Dense Input
python run.py \
    --evaluate=best_epoch.bin \
    --checkpoint=./checkpoints/mixste_gt_reproduced_f_81 \
    --log=log/3dpw_occ_mixste \
    --batch-size=81 \
    -f 81 \
    --keypoints=gt_contiguous_161j \
    -gpu=4 \
    --eval-3dpw \
    --data-path-3dpw=data/h36m_format_for_lifting_3dpw_occ_cpn_dense.npz \
    --only_17 \
    --norm-scale=1.2