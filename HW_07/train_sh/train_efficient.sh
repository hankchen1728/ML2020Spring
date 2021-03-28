DATA_DIR=${1:-"./data/food-11/"}
python3 train.py \
    --data_dir $DATA_DIR \
    --visible_gpus 2 \
    --epochs 200 \
    --lr 0.001 \
    --net_name EfficientNet
