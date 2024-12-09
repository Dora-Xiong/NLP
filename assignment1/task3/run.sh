#!/bin/bash

# 训练模型的参数
BATCH_SIZE=64
UNITS=50
EPOCHS=30
LEARNING_RATE=0.001
TRAIN_FILE="train.txt"
VAL_FILE="val.txt"
TEST_FILE="test.txt"
ENG_VOCAB="eng_bpe.vocab"
JPN_VOCAB="jpn_bpe.vocab"
ENG_KV="eng_word2vec.kv"
JPN_KV="jpn_word2vec.kv"
ENG_MODEL="eng_bpe.model"
JPN_MODEL="jpn_bpe.model"
SAVE_DIR="models"
ENCODER_PATH="models/encoder_epoch9.pth"
DECODER_PATH="models/decoder_epoch9.pth"

# #运行训练脚本
# python lstm_with_attention.py \
#   --batch_size $BATCH_SIZE \
#   --units $UNITS \
#   --epochs $EPOCHS \
#   --learning_rate $LEARNING_RATE \
#   --train_file $TRAIN_FILE \
#   --val_file $VAL_FILE \
#   --test_file $TEST_FILE \
#   --eng_vocab $ENG_VOCAB \
#   --jpn_vocab $JPN_VOCAB \
#   --eng_kv $ENG_KV \
#   --jpn_kv $JPN_KV \
#   --eng_model $ENG_MODEL \
#   --jpn_model $JPN_MODEL \
#   --save_dir $SAVE_DIR

# 如果需要测试特定模型，使用下面的行
python lstm_with_attention.py \
  --test_file $TEST_FILE \
  --eng_vocab $ENG_VOCAB \
  --jpn_vocab $JPN_VOCAB \
  --eng_kv $ENG_KV \
  --jpn_kv $JPN_KV \
  --eng_model $ENG_MODEL \
  --jpn_model $JPN_MODEL \
  --save_dir $SAVE_DIR \
  --test \
  --encoder_path $ENCODER_PATH \
  --decoder_path $DECODER_PATH

