DATA_DIR=${1:-"./data/food-11/"}
PRED_FILE=${2:-"./submission.csv"}

python3 test.py \
    --param_npz ./model_weights/model_wq.npz \
    --data_dir $DATA_DIR \
    --save_path $PRED_FILE \
