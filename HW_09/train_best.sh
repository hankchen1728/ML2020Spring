TRAINX_NPY=${1:-"./data/trainX.npy"}
CKPT_PATH=${2:-"./checkpoints/best.pth"}

python3 train.py \
    --trainX_fpath $TRAINX_NPY \
    --ckpt_path $CKPT_PATH \
    --random_seed 10 \
    --improved \
    --lr 0.0001 \
    --epochs 150
