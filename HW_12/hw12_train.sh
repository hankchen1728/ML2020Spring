python3 train_DaNN.py \
    --visible_gpus 5 \
    --ckpt_dir ./checkpoints \
    --data_dir ./data/real_or_drawing/ \
    --lr 0.001 \
    --epochs 300

python3 train_semi.py \
    --visible_gpus 5 \
    --ckpt_dir ./checkpoints/ \
    --data_dir ./data/real_or_drawing/ \
    --lr 0.0001 \
    --epochs 50
