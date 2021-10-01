#!/bin/bash

# Dir with train/dev data
DATA=data
# The directory to write models/intermediate data
OURDIR=results/lstm-bidir-attn-no-smooth
# For naming output dir, and finding train/dev files
LANG=spanish
# Setting is the size of training set. This sample just has low.
SETTING=low

# Can also call from the module: python -m lstm_inflector.train
python -u lstm_inflector/train.py \
  --train_data_path $DATA/${LANG}-train-${SETTING} \
  --dev_data_path $DATA/${LANG}-dev \
  --output_path $OURDIR \
  --dataset sigmorphon_data \
  --lang ${LANG} \
  --num_epochs 30 \
  --patience 3 \
  --learning_rate 0.001 \
  --batch_size 64 \
  --eval_batch_size 16 \
  --embedding_size  100 \
  --hidden_size 100 \
  --eval_every 1 \
  --eval_after 0 \
  --gradient_clip 1.0 \
  --amsgrad \
	--saveall ;
