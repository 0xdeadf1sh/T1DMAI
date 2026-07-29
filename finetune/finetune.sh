#!/bin/bash

# python finetune.py --checkpoint large.pt --holdout 591 --steps 700 --batch-size 32 --lr-scale 0.2 --warmup 60 --eval-interval 100 --device cuda --dataset ohiot1dm --seed 2

python finetune_multi.py --checkpoint large.pt --steps 5000 --batch-size 32 --lr-scale 0.2 --warmup 150 --eval-interval 100 --device cuda --seed 0
