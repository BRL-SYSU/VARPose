CUDA_VISIBLE_DEVICES=0 python run.py --trainer=decoder \
    --data_path=data/source_data/msst_17_21_25_48_96_coco_192_384_768.pkl \
    --save_dir=./checkpoints/trainer_decoder_coco_17 \
    --log_dir=./logs/trainer_decoder_coco_17 \
    --gt_patch=coco_17 \
    --checkpoint_path=checkpoints/flexible_granularity_decoder/decoder_coco_17.pth \
    --var_model_path=checkpoints/varpose_ar/var_best.pth \
    --vqvae_model_path=checkpoints/varpose_gpt_hierarchical/vqvae_best.pth \
    --test

CUDA_VISIBLE_DEVICES=0 python run.py --trainer=decoder \
    --data_path=data/source_data/msst_17_21_25_48_96_coco_192_384_768.pkl \
    --save_dir=./checkpoints/trainer_decoder_192 \
    --log_dir=./logs/trainer_decoder_192 \
    --gt_patch=192 \
    --checkpoint_path=checkpoints/flexible_granularity_decoder/decoder_192.pth \
    --var_model_path=checkpoints/varpose_ar/var_best.pth \
    --vqvae_model_path=checkpoints/varpose_gpt_hierarchical/vqvae_best.pth \
    --test

CUDA_VISIBLE_DEVICES=0 python run.py --trainer=decoder \
    --data_path=data/source_data/msst_17_21_25_48_96_coco_192_384_768.pkl \
    --save_dir=./checkpoints/trainer_decoder_384 \
    --log_dir=./logs/trainer_decoder_384 \
    --gt_patch=384 \
    --checkpoint_path=checkpoints/flexible_granularity_decoder/decoder_384.pth \
    --var_model_path=checkpoints/varpose_ar/var_best.pth \
    --vqvae_model_path=checkpoints/varpose_gpt_hierarchical/vqvae_best.pth \
    --test

CUDA_VISIBLE_DEVICES=0 python run.py --trainer=decoder \
    --data_path=data/source_data/msst_17_21_25_48_96_coco_192_384_768.pkl \
    --save_dir=./checkpoints/trainer_decoder_768 \
    --log_dir=./logs/trainer_decoder_768 \
    --gt_patch=768 \
    --checkpoint_path=checkpoints/flexible_granularity_decoder/decoder_768.pth \
    --var_model_path=checkpoints/varpose_ar/var_best.pth \
    --vqvae_model_path=checkpoints/varpose_gpt_hierarchical/vqvae_best.pth \
    --test