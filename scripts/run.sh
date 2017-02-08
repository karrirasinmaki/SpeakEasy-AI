#!/bin/bash
# Runs the speak_easy model.

ROBOT_NAME=MARVIN

venv/bin/python speak_easy.py \
  --num_layers 3 \
  --size 128 \ # 762 \
  --steps_per_checkpoint 500 \
  --batch_size 128 \
  --vocab_size 200000 \
  ---max_train_data_size 0 \
  --train_dir '../vauvafi-crawler/output/train' \
  --data_dir '../vauvafi-crawler/output/data' \
  --log_dir '../vauvafi-crawler/output/log' \
  $@
