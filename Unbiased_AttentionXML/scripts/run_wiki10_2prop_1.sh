#!/bin/bash

#SBATCH --job-name=wiki10_2prop_1
#SBATCH --error=out/wiki10_2prop_1.err
#SBATCH --output=out/wiki10_2prop_1.out
#SBATCH --account=Project_2001083
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:2

module load pytorch/1.0.1


cd ..

DATA=Wiki10-31K
MODEL=AttentionXML
REWEIGHTING=2prop-1
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
