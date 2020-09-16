#!/usr/bin/env bash

set -e

GPU=0
NORM=gn
DS=ogbg-molhiv
LOG_PATH=../../log/Example-gin-test/"$NORM"/
MODEL_PATH=../../model/Example-gin-test/"$NORM"/
DATA_PATH=../../data/dataset/


for seed in {1..5}; do
    FILE_NAME=learn-"$DS"-gin-seed-"$seed" 
    python ../../src/train_dgl_ogb.py \
        --gpu $GPU \
        --seed $seed \
        --dataset $DS \
        --log_dir $LOG_PATH \
        --model_path $MODEL_PATH \
        --data_dir $DATA_PATH \
        --exp $FILE_NAME \
        --norm_type $NORM \
        --log_norm
done

