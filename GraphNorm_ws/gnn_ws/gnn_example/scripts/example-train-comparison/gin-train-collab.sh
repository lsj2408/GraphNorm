#!/usr/bin/env bash

set -e
GPU=0
NORM=gn
MODEL=GIN
DS=COLLAB
EXP_NAME="DS"-train-comparison-gin
LOG_PATH=../../../log/Example-train-comparison/gin-reproduce/"$DS"/"$NORM"/
DATA_PATH=../../../data/

python ../../train_graph_level.py \
    --gpu $GPU \
    --dataset $DS \
    --exp $EXP_NAME \
    --log_dir $LOG_PATH \
    --data_dir $DATA_PATH \
    --model $MODEL \
    --log_norm \
    --norm_type $NORM \
    --graph_pooling_type mean \
    --cross_validation \
    --degree_as_label \
    --self_loop 
