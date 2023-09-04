<<<<<<< HEAD
CUDA_VISIBLE_DEVICES=4,6 python -u /mnt/cephfs/home/alvin/wenfu/project/SlowFast-prompt/tools/run_net.py \
    --checkpoint_path /mnt/cephfs/dataset/m3lab_data/wf_data/stts_pth/ek100_log/all_mv2S_10pmpt_s06_fc_bs24_lr1e-7_epic \
    --cfg /mnt/cephfs/home/alvin/wenfu/project/SlowFast-prompt/configs/epick/MVITv2_S_16x4_epick.yaml \
=======
CUDA_VISIBLE_DEVICES=4,5,6,7 python -u /mnt/cephfs/home/alvin/wenfu/coderepo/SlowFast-main/tools/run_net.py \
    --checkpoint_path /mnt/cephfs/dataset/m3lab_data/wf_data/stts_pth/exp_log/mv2_S_10pmpt_0909_fc_bs24_lr1e-7_epic \
    --cfg /mnt/cephfs/home/alvin/wenfu/coderepo/SlowFast-main/configs/epick/MVITv2_S_16x4_epick.yaml \
>>>>>>> 857e6de891297b3edcacb69afdd3ff81944be336
    TRAIN.ENABLE True \
    TRAIN.BATCH_SIZE 24 \
    NUM_GPUS 2 \
    TEST.ENABLE True \
    MIXUP.ENABLE True \
    SOLVER.MAX_EPOCH 50 \
<<<<<<< HEAD
    OUTPUT_DIR /mnt/cephfs/dataset/m3lab_data/wf_data/stts_pth/ek100_log/all_mv2S_10pmpt_s06_fc_bs24_lr1e-7_epic \
    TEST.CHECKPOINT_FILE_PATH /mnt/cephfs/dataset/m3lab_data/wf_data/stts_pth/ek100_log/all_mv2S_10pmpt_s06_fc_bs24_lr1e-7_epic/checkpoints/best_checkpoint.pyth \
    TEST.BATCH_SIZE 64 \
=======
    OUTPUT_DIR /mnt/cephfs/dataset/m3lab_data/wf_data/stts_pth/exp_log/mv2_S_10pmpt_0909_fc_bs24_lr1e-7_epic \
    TEST.CHECKPOINT_FILE_PATH /mnt/cephfs/dataset/m3lab_data/wf_data/stts_pth/exp_log/mv2_S_10pmpt_0909_fc_bs24_lr1e-7_epic/checkpoints/best_checkpoint.pyth \
    TEST.BATCH_SIZE 32 \
>>>>>>> 857e6de891297b3edcacb69afdd3ff81944be336
    SOLVER.BASE_LR 0.0001 \
    SOLVER.COSINE_END_LR 1e-7 \
    SOLVER.WARMUP_START_LR 1e-7 \
    SOLVER.WARMUP_EPOCHS 10.0 \
    SOLVER.LR_POLICY cosine \
<<<<<<< HEAD
    MVIT.NUM_PROMPT 10 \
    MVIT.TEMPORAL_LOC None \
    MVIT.SPATIAL_LOC [4] \
    MVIT.TIME_LEFT_RATIO [0.6] \
    MVIT.SPACE_LEFT_RATIO [0.6] 
=======
    MVIT.TEMPORAL_LOC [0] \
    MVIT.SPATIAL_LOC [4] \
    MVIT.TIME_LEFT_RATIO [0.9] \
    MVIT.SPACE_LEFT_RATIO [0.9]
>>>>>>> 857e6de891297b3edcacb69afdd3ff81944be336
