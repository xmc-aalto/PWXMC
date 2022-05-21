#!/usr/bin/env bash


DATA=Amazon-670K
MODEL=FastAttentionXML
REWEIGHTING=PW
A=0.6
B=2.6

./scripts/run_preprocess.sh $DATA
./scripts/run_xml.sh $DATA $MODEL $REWEIGHTING $A $B

python evaluation.py \
--results results/$MODEL-$DATA-reweighting-$REWEIGHTING-Ensemble-labels.npy \
--targets data/$DATA/test_labels.npy \
--train-labels data/$DATA/train_labels.npy \
-a $A \
-b $B
