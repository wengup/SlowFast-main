CUDA_VISIBLE_DEVICES=4,5,0 python -u /mnt/cephfs/home/alvin/wenfu/coderepo/SlowFast-main/tools/run_net.py \
    --cfg /mnt/cephfs/home/alvin/wenfu/coderepo/SlowFast-main/configs/epick/SLOWFAST_8x8_R50_epick.yaml \
    TRAIN.ENABLE True \
    TRAIN.BATCH_SIZE 12 \
    NUM_GPUS 3 \
    TEST.ENABLE False \
    MIXUP.ENABLE False \
    SOLVER.MAX_EPOCH 50 \
    OUTPUT_DIR /mnt/cephfs/dataset/wf_data/stts_pth/output/SF8x8_R50_epic \
    TEST.BATCH_SIZE 60 \

