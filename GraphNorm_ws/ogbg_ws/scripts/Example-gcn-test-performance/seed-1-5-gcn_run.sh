#!/usr/bin/env bash

set -e

GPU=2
NORM=gn
DS=ogbg-molhiv
LOG_PATH=../../log/Example-gcn-test/"$NORM"/
MODEL_PATH=../../model/Example-gcn-test/"$NORM"/
DATA_PATH=../../data/dataset/


for seed in {1..5}; do
    FILE_NAME=learn-"$DS"-gcn-seed-"$seed"
    python ../../src/train_dgl_ogb.py \
        --gpu $GPU \
        --epoch 50 \
        --model GCN \
        --batch_size 64 \
        --seed $seed \
        --dataset $DS \
        --log_dir $LOG_PATH \
        --model_path $MODEL_PATH \
        --data_dir $DATA_PATH \
        --exp $FILE_NAME \
        --norm_type $NORM \
        --log_norm
done

