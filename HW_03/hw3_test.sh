DATA_DIR=${1:-"./data/food-11/"}
PRED_FILE=${2:-"./submission.csv"}

python3 test.py \
    --ckpt_path ./model_weights/EfficientNetB3_best.pth \
    --data_dir $DATA_DIR \
    --save_path $PRED_FILE \
    # --net_name EfficientNetB3
