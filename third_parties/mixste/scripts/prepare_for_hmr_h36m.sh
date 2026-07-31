python run.py \
    --evaluate=best_epoch.bin \
    --checkpoint=./checkpoints/mixste_concat_gt_f_81 \
    --log=log/mixste_varpose_output \
    --batch-size=81 \
    -f 81 \
    --keypoints=gt_contiguous_161j \
    --save-prediction=output/mixste_concat_gt_f_81.npz \
    -gpu=1

python prepare_for_mesh_eval_h36m.py \
    --prediction output/mixste_concat_gt_f_81.npz \
    --pkl data/msst_data_h36m_vp3d_standard.pkl \
    --output output/mixste_concat_gt_f_81_integrated.npz