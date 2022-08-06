#!/usr/bin/env bash

#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 06 August, 2022
#Run every script in the NNCLR tests


echo "Experiment 1"
python3 -Wignore src/experiment1.py > ex1_log.txt

echo "Experiment 2"
python3 -Wignore src/experiment1.py > ex2_log.txt

echo "Experiment 3"
python3 -Wignore src/experiment1.py > ex3_log.txt