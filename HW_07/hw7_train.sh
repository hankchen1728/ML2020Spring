DATA_DIR=${1:-"./data/food-11/"}

python3 train.py \
    --data_dir $DATA_DIR \
    --epochs 200 \
    --net_name MobileNetV2
