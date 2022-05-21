#!/bin/bash

python main.py \
--model_type xlnet \
--model_name_or_path pretrained_models/xlnet_base_cased \
--task_name Wiki500k \
--do_train \
--do_eval \
--eval_all_checkpoints \
--overwrite_output_dir \
--data_dir ../data/Wiki500k \
--max_seq_length 256 \
--per_gpu_train_batch_size=16 \
--per_gpu_eval_batch_size=32 \
--learning_rate_x 5e-5 \
--learning_rate_h 1e-4 \
--learning_rate_a 2e-3 \
--num_train_epochs 12.0 \
--output_dir ../models/Wiki500k \
--pos_label 274 \
--adaptive_cutoff 167000 334000 \
--div_value 2 \
--logging_steps 50000 \
--save_steps 50000 \
--gpu 0,1,2,3 \
--reweighting PW \
--A 0.5 \
--B 0.4



