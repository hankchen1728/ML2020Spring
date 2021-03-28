DATA_DIR=${1:-"./data/faces/"}
GEN_CKPT=${2:-"./checkpoints/p2_g.th"}

python3 train_WGAN.py \
    --gen_ckpt $GEN_CKPT \
    --log_dir "./log/WGAN" \
    --data_dir $DATA_DIR \
    --lr 0.0001 \
    --epochs 10
