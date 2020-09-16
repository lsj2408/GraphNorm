#!/usr/bin/env bash

set -e

# DS [PROTEINS, NCI1, MUTAG, PTC] for bioinformatics datasets
GPU=0
NORM=gn
MODEL=GIN
DS=PROTEINS
EXP_NAME="DS"-test-comparison-gin
LOG_PATH=../../../log/Example-test-comparison/gin-reproduce/"$DS"/"$NORM"/
DATA_PATH=../../../data/

python ../../train_graph_level.py \
    --gpu $GPU \
    --dataset $DS \
    --exp $EXP_NAME \
    --log_dir $LOG_PATH \
    --data_dir $DATA_PATH \
    --model $MODEL \
    --weight_decay 0.05 \
    --log_norm \
    --norm_type $NORM \
    --graph_pooling_type sum \
    --cross_validation \
    --self_loop \
    --learn_eps 
