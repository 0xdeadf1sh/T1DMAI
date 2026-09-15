#!/bin/bash

python finetune_omar.py --checkpoint models/d16_dilate/checkpoints/t1dmai_best.pt --total-steps 2000 --batch-size 64 --num-workers 4 --cache datasets/t1dmdroid/t1dm-20260915-1407/ --seed 42 --eval-seed 42 --validation-interval 100 --warmup-steps 200
