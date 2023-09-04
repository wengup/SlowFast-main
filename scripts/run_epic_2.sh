CUDA_VISIBLE_DEVICES=2,3,4,5 python -u /mnt/cephfs/home/alvin/wenfu/project/SlowFast-prompt/tools/run_net.py \
    --checkpoint_path /mnt/cephfs/dataset/m3lab_data/wf_data/stts_pth/mini_EK100/mv2S_10pmpt_s08_fc_bs24_lr1e-7_epic \
    --cfg /mnt/cephfs/home/alvin/wenfu/project/SlowFast-prompt/configs/epick/MVITv2_S_16x4_epick.yaml \
    TRAIN.ENABLE True \
    TRAIN.BATCH_SIZE 24 \
    NUM_GPUS 4 \
    TEST.ENABLE True \
    MIXUP.ENABLE False \
    SOLVER.MAX_EPOCH 50 \
    OUTPUT_DIR /mnt/cephfs/dataset/m3lab_data/wf_data/stts_pth/mini_EK100/mv2S_10pmpt_s08_fc_bs24_lr1e-7_epic \
    TEST.CHECKPOINT_FILE_PATH /mnt/cephfs/dataset/m3lab_data/wf_data/stts_pth/mini_EK100/mv2S_10pmpt_s08_fc_bs24_lr1e-7_epic/checkpoints/best_checkpoint.pyth \
    TEST.BATCH_SIZE 32 \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.COSINE_END_LR 1e-7 \
    SOLVER.WARMUP_START_LR 1e-7 \
    SOLVER.WARMUP_EPOCHS 30.0 \
    SOLVER.LR_POLICY cosine \
    MVIT.NUM_PROMPT 10 \
    MVIT.TEMPORAL_LOC None \
    MVIT.SPATIAL_LOC [4] \
    MVIT.TIME_LEFT_RATIO [0.8] \
    MVIT.SPACE_LEFT_RATIO [0.8]

CUDA_VISIBLE_DEVICES=2,3,4,5 python -u /mnt/cephfs/home/alvin/wenfu/project/SlowFast-prompt/tools/run_net.py \
    --checkpoint_path /mnt/cephfs/dataset/m3lab_data/wf_data/stts_pth/mini_EK100/mv2S_10pmpt_s06_fc_bs24_lr1e-7_epic \
    --cfg /mnt/cephfs/home/alvin/wenfu/project/SlowFast-prompt/configs/epick/MVITv2_S_16x4_epick.yaml \
    TRAIN.ENABLE True \
    TRAIN.BATCH_SIZE 24 \
    NUM_GPUS 4 \
    TEST.ENABLE True \
    MIXUP.ENABLE False \
    SOLVER.MAX_EPOCH 50 \
    OUTPUT_DIR /mnt/cephfs/dataset/m3lab_data/wf_data/stts_pth/mini_EK100/mv2S_10pmpt_s06_fc_bs24_lr1e-7_epic \
    TEST.CHECKPOINT_FILE_PATH /mnt/cephfs/dataset/m3lab_data/wf_data/stts_pth/mini_EK100/mv2S_10pmpt_s06_fc_bs24_lr1e-7_epic/checkpoints/best_checkpoint.pyth \
    TEST.BATCH_SIZE 32 \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.COSINE_END_LR 1e-7 \
    SOLVER.WARMUP_START_LR 1e-7 \
    SOLVER.WARMUP_EPOCHS 30.0 \
    SOLVER.LR_POLICY cosine \
    MVIT.NUM_PROMPT 10 \
    MVIT.TEMPORAL_LOC None \
    MVIT.SPATIAL_LOC [4] \
    MVIT.TIME_LEFT_RATIO [0.8] \
    MVIT.SPACE_LEFT_RATIO [0.6]

CUDA_VISIBLE_DEVICES=2,3,4,5 python -u /mnt/cephfs/home/alvin/wenfu/project/SlowFast-prompt/tools/run_net.py \
    --checkpoint_path /mnt/cephfs/dataset/m3lab_data/wf_data/stts_pth/mini_EK100/mv2S_10pmpt_t09s09_fc_bs24_lr1e-7_epic \
    --cfg /mnt/cephfs/home/alvin/wenfu/project/SlowFast-prompt/configs/epick/MVITv2_S_16x4_epick.yaml \
    TRAIN.ENABLE True \
    TRAIN.BATCH_SIZE 24 \
    NUM_GPUS 4 \
    TEST.ENABLE True \
    MIXUP.ENABLE False \
    SOLVER.MAX_EPOCH 50 \
    OUTPUT_DIR /mnt/cephfs/dataset/m3lab_data/wf_data/stts_pth/mini_EK100/mv2S_10pmpt_t09s09_fc_bs24_lr1e-7_epic \
    TEST.CHECKPOINT_FILE_PATH /mnt/cephfs/dataset/m3lab_data/wf_data/stts_pth/mini_EK100/mv2S_10pmpt_t09s09_fc_bs24_lr1e-7_epic/checkpoints/best_checkpoint.pyth \
    TEST.BATCH_SIZE 32 \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.COSINE_END_LR 1e-7 \
    SOLVER.WARMUP_START_LR 1e-7 \
    SOLVER.WARMUP_EPOCHS 30.0 \
    SOLVER.LR_POLICY cosine \
    MVIT.NUM_PROMPT 10 \
    MVIT.TEMPORAL_LOC [0] \
    MVIT.SPATIAL_LOC [4] \
    MVIT.TIME_LEFT_RATIO [0.9] \
    MVIT.SPACE_LEFT_RATIO [0.9]
