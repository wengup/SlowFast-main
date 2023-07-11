CUDA_VISIBLE_DEVICES=7,4,3,2,5,6 python -u /mnt/cephfs/home/alvin/wenfu/coderepo/SlowFast-main/tools/run_net.py \
    --cfg /mnt/cephfs/home/alvin/wenfu/coderepo/SlowFast-main/configs/epick/MVITv2_B_32x3_epick.yaml \
    TRAIN.ENABLE True \
    TRAIN.BATCH_SIZE 12 \
    NUM_GPUS 6 \
    TEST.ENABLE False \
    MIXUP.ENABLE False \
    SOLVER.MAX_EPOCH 50 \
    OUTPUT_DIR /mnt/cephfs/dataset/wf_data/stts_pth/output/mvitv2_B_epic \
    TEST.BATCH_SIZE 60 \

