#!/usr/bin/env bash

set -e
GPU=1
NORM=gn
MODEL=GIN
DS=REDDITBINARY
EXP_NAME="DS"-test-comparison-gin
LOG_PATH=../../../log/Example-test-comparison/gin-reproduce/"$DS"/"$NORM"/
DATA_PATH=../../../data/

python ../../train_graph_level.py \
    --gpu $GPU \
    --lr 0.001 \
    --dataset $DS \
    --exp $EXP_NAME \
    --log_dir $LOG_PATH \
    --data_dir $DATA_PATH \
    --model $MODEL \
    --log_norm \
    --norm_type $NORM \
    --graph_pooling_type mean \
    --cross_validation \
    --self_loop \
    --learn_eps
