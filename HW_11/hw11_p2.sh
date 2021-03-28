CKPT_PATH=${1:-"./checkpoints/p2_g.pth"}
OUT_IMG=${2:-"./image_rep/p2.png"}

python3 test_WGAN.py \
    --gen_ckpt $CKPT_PATH \
    --save_img $OUT_IMG
