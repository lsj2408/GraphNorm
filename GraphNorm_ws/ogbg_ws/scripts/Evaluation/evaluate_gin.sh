#!/usr/bin/env bash

set -e

GPU=0
NORM=gn
MODEL=GIN
DS=ogbg-molhiv
BS=128
MODEL_PATH=../../model/Pre-train-model/GIN-GN/
DATA_PATH=../../data/dataset/


python ../../src/evaluate_ogb.py \
    --gpu $GPU \
    --dataset $DS \
    --model_path $MODEL_PATH \
    --data_dir $DATA_PATH \
    --norm_type $NORM \
    --model $MODEL \
    --batch_size $BS 

