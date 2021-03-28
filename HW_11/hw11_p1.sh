CKPT_PATH=${1:-"./checkpoints/p1_g.pth"}
OUT_IMG=${2:-"./image_rep/p1.png"}

python3 test_GAN.py \
    --gen_ckpt $CKPT_PATH \
    --save_img $OUT_IMG

