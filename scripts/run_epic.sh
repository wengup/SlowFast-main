CUDA_VISIBLE_DEVICES=0,3 python -u /mnt/cephfs/home/alvin/wenfu/coderepo/STTS/MViT/tools/run_net.py \
    --cfg /mnt/cephfs/home/alvin/wenfu/coderepo/STTS/MViT/configs/epick/epic_kitchens_224_32x3.yaml \
    TRAIN.ENABLE True \
    TRAIN.BATCH_SIZE 16 \
    MVIT.TIME_PRUNING_LOC None \
    MVIT.SPACE_PRUNING_LOC None \
    MODEL.NUM_CLASSES 8 \
    NUM_GPUS 2 \
    TEST.ENABLE True \
    MIXUP.ENABLE False \
    SOLVER.MAX_EPOCH 50 \
    DATA.NUM_FRAMES 16 \
    OUTPUT_DIR /mnt/cephfs/dataset/wf_data/stts_pth/output/no_pretained \
    TEST.BATCH_SIZE 60

