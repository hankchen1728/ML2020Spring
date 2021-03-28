TRAIN_FPATH=${1:-"./data/training_label.txt"}
TRAIN_NOLABEL_FPATH=${2:-"./data/training_nolabel.txt"}

python3 train.py \
    --train_fpath $TRAIN_FPATH \
    --train_nolabel_fpath $TRAIN_NOLABEL_FPATH \
    --ckpt_dir "./checkpoint" \
    --w2v_path "./checkpoint/w2v_all_100.model" \
    --max_sen_len 30 \
    --lr 0.001 \
    --epochs 20
