TRAINX_NPY=${1:-"./data/trainX.npy"}
CKPT_PATH=${2:-"./checkpoints/best.pth"}
PRED_CSV=${3:-"./reproduce.csv"}

python3 image_cluster.py \
    --trainX_fpath $TRAINX_NPY \
    --ckpt_path $CKPT_PATH \
    --predict_csv $PRED_CSV \
    --train_clustering \
    --pca_ndim 100 \
    --random_seed 10
