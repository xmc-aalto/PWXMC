#!/usr/bin/env bash

python main.py --data-cnf configure/datasets/$1.yaml --model-cnf configure/models/$2-$1.yaml --reweighting $3 -a $4 -b $5 -t 0
python main.py --data-cnf configure/datasets/$1.yaml --model-cnf configure/models/$2-$1.yaml --reweighting $3 -a $4 -b $5 -t 1
python main.py --data-cnf configure/datasets/$1.yaml --model-cnf configure/models/$2-$1.yaml --reweighting $3 -a $4 -b $5 -t 2
python ensemble.py -p results/$2-$1-reweighting-$3 -t 3
