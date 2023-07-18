CUDA_VISIBLE_DEVICES=4,5,6,7 python -u /mnt/cephfs/home/alvin/wenfu/coderepo/SlowFast-main/tools/run_net.py \
    --checkpoint_path /mnt/cephfs/dataset/m3lab_data/wf_data/stts_pth/exp_log/mv2_S_t06s09_bs24_lr1e-7_epic/test \
    --cfg /mnt/cephfs/home/alvin/wenfu/coderepo/SlowFast-main/configs/epick/MVITv2_S_16x4_epick.yaml \
    TRAIN.ENABLE False \
    TRAIN.BATCH_SIZE 24 \
    NUM_GPUS 4 \
    TEST.ENABLE True \
    MIXUP.ENABLE False \
    SOLVER.MAX_EPOCH 50 \
    OUTPUT_DIR /mnt/cephfs/dataset/m3lab_data/wf_data/stts_pth/exp_log/mv2_S_t06s09_bs24_lr1e-7_epic/test \
    TEST.CHECKPOINT_FILE_PATH /mnt/cephfs/dataset/m3lab_data/wf_data/stts_pth/exp_log/mv2_S_t06s09_bs24_lr1e-7_epic/checkpoints/best_checkpoint.pyth \
    TEST.BATCH_SIZE 32 \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.COSINE_END_LR 1e-7 \
    SOLVER.WARMUP_START_LR 1e-7 \
    SOLVER.WARMUP_EPOCHS 30.0 \
    SOLVER.LR_POLICY cosine \
    MVIT.TIME_PRUNING_LOC [0] \
    MVIT.SPACE_PRUNING_LOC [4] \
    MVIT.TIME_LEFT_RATIO [0.6] \
    MVIT.SPACE_LEFT_RATIO [0.9] \

CUDA_VISIBLE_DEVICES=4,5,6,7 python -u /mnt/cephfs/home/alvin/wenfu/coderepo/SlowFast-main/tools/run_net.py \
    --checkpoint_path /mnt/cephfs/dataset/m3lab_data/wf_data/stts_pth/exp_log/mv2_S_t05s07_bs24_lr1e-7_epic/test \
    --cfg /mnt/cephfs/home/alvin/wenfu/coderepo/SlowFast-main/configs/epick/MVITv2_S_16x4_epick.yaml \
    TRAIN.ENABLE False \
    TRAIN.BATCH_SIZE 24 \
    NUM_GPUS 4 \
    TEST.ENABLE True \
    MIXUP.ENABLE False \
    SOLVER.MAX_EPOCH 50 \
    OUTPUT_DIR /mnt/cephfs/dataset/m3lab_data/wf_data/stts_pth/exp_log/mv2_S_t05s07_bs24_lr1e-7_epic/test \
    TEST.CHECKPOINT_FILE_PATH /mnt/cephfs/dataset/m3lab_data/wf_data/stts_pth/exp_log/mv2_S_t05s07_bs24_lr1e-7_epic/checkpoints/best_checkpoint.pyth \
    TEST.BATCH_SIZE 32 \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.COSINE_END_LR 1e-7 \
    SOLVER.WARMUP_START_LR 1e-7 \
    SOLVER.WARMUP_EPOCHS 30.0 \
    SOLVER.LR_POLICY cosine \
    MVIT.TIME_PRUNING_LOC [0] \
    MVIT.SPACE_PRUNING_LOC [4] \
    MVIT.TIME_LEFT_RATIO [0.5] \
    MVIT.SPACE_LEFT_RATIO [0.7] \

CUDA_VISIBLE_DEVICES=4,5,6,7 python -u /mnt/cephfs/home/alvin/wenfu/coderepo/SlowFast-main/tools/run_net.py \
    --checkpoint_path /mnt/cephfs/dataset/m3lab_data/wf_data/stts_pth/exp_log/mv2_S_t03s08_bs24_lr1e-7_epic/test \
    --cfg /mnt/cephfs/home/alvin/wenfu/coderepo/SlowFast-main/configs/epick/MVITv2_S_16x4_epick.yaml \
    TRAIN.ENABLE False \
    TRAIN.BATCH_SIZE 24 \
    NUM_GPUS 4 \
    TEST.ENABLE True \
    MIXUP.ENABLE False \
    SOLVER.MAX_EPOCH 50 \
    OUTPUT_DIR /mnt/cephfs/dataset/m3lab_data/wf_data/stts_pth/exp_log/mv2_S_t03s08_bs24_lr1e-7_epic/test \
    TEST.CHECKPOINT_FILE_PATH /mnt/cephfs/dataset/m3lab_data/wf_data/stts_pth/exp_log/mv2_S_t03s08_bs24_lr1e-7_epic/checkpoints/best_checkpoint.pyth \
    TEST.BATCH_SIZE 32 \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.COSINE_END_LR 1e-7 \
    SOLVER.WARMUP_START_LR 1e-7 \
    SOLVER.WARMUP_EPOCHS 30.0 \
    SOLVER.LR_POLICY cosine \
    MVIT.TIME_PRUNING_LOC [0] \
    MVIT.SPACE_PRUNING_LOC [4] \
    MVIT.TIME_LEFT_RATIO [0.3] \
    MVIT.SPACE_LEFT_RATIO [0.8] \

