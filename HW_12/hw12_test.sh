DATA_DIR=${1:-"./data/real_or_drawing/"}
PRED_FILE=${2:-"./submission.csv"}

python3 test.py \
    --data_dir $DATA_DIR \
    --ckpt_dir "./checkpoints/" \
    --save_path $PRED_FILE
