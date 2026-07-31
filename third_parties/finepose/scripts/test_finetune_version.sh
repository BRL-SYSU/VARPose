python main.py \
    --finetune \
    -c checkpoints/cpn_dense \
    --evaluate best_epoch_32_40.79.bin \
    -num_proposals 20 \
    -sampling_timesteps 10 \
    -b 2048 \
    -gpu 0,1,2,3,4,5,6,7 \
    --nolog