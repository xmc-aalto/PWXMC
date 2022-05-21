#!/bin/bash

#SBATCH --job-name=wiki500_2prop_1_tree_1
#SBATCH --error=out/wiki500_2prop_1_tree_1.err
#SBATCH --output=out/wiki500_2prop_1_tree_1.out
#SBATCH --account=Project_2001083
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=100G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:4

module load pytorch/1.0.1

cd ..


DATA=Wiki-500K
MODEL=FastAttentionXML
REWEIGHTING=2prop-1
A=0.5
B=0.4

#./scripts/run_preprocess.sh $DATA
./scripts/run_xml_tree_1.sh $DATA $MODEL $REWEIGHTING $A $B

python evaluation.py \
--results results/$MODEL-$DATA-reweighting-$REWEIGHTING-Ensemble-labels.npy \
--targets data/$DATA/test_labels.npy \
--train-labels data/$DATA/train_labels.npy
