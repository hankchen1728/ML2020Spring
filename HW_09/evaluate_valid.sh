VALX_NPY=${1:-"./data/valX.npy"}
CKPT_PATH=${2:-"./checkpoints/improved.pth"}
PRED_CSV=${3:-"./valid_predict.csv"}

python3 image_cluster.py \
    --visible_gpus 5 \
    --trainX_fpath $VALX_NPY \
    --ckpt_path $CKPT_PATH \
    --predict_csv $PRED_CSV \
    --train_clustering \
    --label_fpath "./data/valY.npy" \
    --sklearn_model_base "./checkpoints/" \
    --random_seed 10
