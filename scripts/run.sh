#!/bin/bash
# Runs the speak_easy model.

TRAIN_DIR='../vauvafi-crawler/output/train'
DATA_DIR='../vauvafi-crawler/output/data'
LOG_DIR='../vauvafi-crawler/output/log'
ROBOT_NAME=MARVIN


NUM_LAYERS=3
SIZE=762
VOCAB_SIZE=1000
MAX_TRAIN_DATA_SIZE=0

venv/bin/python speak_easy.py --num_layers $NUM_LAYERS --size $SIZE --vocab_size $VOCAB_SIZE ---max_train_data_size $MAX_TRAIN_DATA_SIZE --train_dir $TRAIN_DIR --data_dir $DATA_DIR --log_dir $LOG_DIR $@
