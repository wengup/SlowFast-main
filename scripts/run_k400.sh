CUDA_VISIBLE_DEVICES=4,7,6 python -u /mnt/cephfs/home/alvin/yangzehang/workplace/slowfast/tools/run_net.py \
    --cfg /mnt/cephfs/home/alvin/wenfu/coderepo/SlowFast-main/configs/Kinetics/MVITv2_B_32x3.yaml \
    TRAIN.ENABLE False \
    TRAIN.BATCH_SIZE 16 \
    NUM_GPUS 3 \
    TEST.ENABLE True \
    OUTPUT_DIR /mnt/cephfs/dataset/wf_data/stts_pth/output/mvitv2_k400 \
    TEST.BATCH_SIZE 60 \
    TEST.CHECKPOINT_FILE_PATH /mnt/cephfs/dataset/wf_data/stts_pth/MViTv2_B_32x3_k400_f304025456.pyth \
    DATA.PATH_TO_DATA_DIR /mnt/cephfs/dataset/wf_data/stts_pth/k400_annotations

