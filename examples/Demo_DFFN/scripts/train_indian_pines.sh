#!/bin/bash

rm -rf /scratch0/ilya/locDoc/caffe-tensorflow/models/throw;

CUDA_VISIBLE_DEVICES=1 \
python main.py \
--dataset IP \
--eval_period 10 \
--num_epochs 100000 \
--mask_root  /scratch0/ilya/locDoc/data/hyperspec/Indian_pines_gt_traintest_ma2015_1_9146f0.mat \
--model_root /scratch0/ilya/locDoc/caffe-tensorflow/models/throw 
