#!/usr/bin/env bash

set -e

# DS [PROTEINS, NCI1, MUTAG, PTC] for bioinformatics datasets
GPU=2
NORM=gn
MODEL=GCN
DS=NCI1
EXP_NAME="DS"-train-comparison-gcn
LOG_PATH=../../../log/Example-train-comparison/gcn-reproduce/"$DS"/"$NORM"/
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
    --graph_pooling_type sum \
    --cross_validation \
    --self_loop
