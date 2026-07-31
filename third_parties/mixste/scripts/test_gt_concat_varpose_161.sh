python run.py \
    --evaluate=best_epoch.bin \
    --checkpoint=./checkpoints/mixste_concat_gt_f_81 \
    --log=log/evaluate_25.0 \
    --batch-size=81 \
    -f 81 \
    --keypoints=gt_contiguous_161j \
    -gpu=1