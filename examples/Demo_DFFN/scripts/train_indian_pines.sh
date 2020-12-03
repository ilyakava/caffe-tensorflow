#!/bin/bash

rm -rf /scratch0/ilya/locDoc/caffe-tensorflow/models/throw;

CUDA_VISIBLE_DEVICES=1 \
python main.py \
--dataset IP \
--eval_period 10 \
--num_epochs 100000 \
--mask_root  /scratch0/ilya/locDoc/data/hyperspec/Indian_pines_gt_traintest_p05_1_f0b0f8.mat \
--model_root /scratch0/ilya/locDoc/caffe-tensorflow/models/throw 

# Indian_pines_gt_traintest_p02_1_2c6de4.mat