#!/usr/bin/env bash


DATA=AmazonCat-13K
MODEL=AttentionXML
REWEIGHTING=PW
A=0.55
B=1.5

./scripts/run_preprocess.sh $DATA
./scripts/run_xml.sh $DATA $MODEL $REWEIGHTING $A $B

python evaluation.py \
--results results/$MODEL-$DATA-reweighting-$REWEIGHTING-Ensemble-labels.npy \
--targets data/$DATA/test_labels.npy \
--train-labels data/$DATA/train_labels.npy \
-a $A \
-b $B
