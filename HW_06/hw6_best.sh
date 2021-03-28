INPUT_DIR=${1:-"./data/"}
OUTPUT_IMG_DIR=${2:-"./output/"}

python3 fgsm.py \
    --data_dir $INPUT_DIR \
    --save_dir $OUTPUT_IMG_DIR \
    --net_name densenet121 \
    --iter 2 \
    --epsilon 0.05
