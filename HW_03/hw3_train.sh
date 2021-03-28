DATA_DIR=${1:-"./data/food-11/"}
python3 train.py \
    --data_dir $DATA_DIR \
    --visible_gpus 0 \
    --epochs 120 \
    --net_name EfficientNet
