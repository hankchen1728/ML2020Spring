TEST_FPATH=${1:-"./data/testing_data.txt"}
SAVE_PATH=${2:-"./submission.csv"}

python3 test.py \
    --test_fpath $TEST_FPATH \
    --ckpt_path "./model_weight/GRU_best.pth" \
    --w2v_path "./model_weight/w2v_all_100.model" \
    --max_sen_len 30 \
    --save_path $SAVE_PATH
