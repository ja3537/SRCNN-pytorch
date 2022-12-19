#!/bin/bash
lr=1e-4

for prev in {3..19}
do
  python3 continue_analog_training.py --prev-epoch $prev --num-workers 0 --train-file data/91-image_x4.h5 --eval-file data/Set5_x4.h5 --scale 4 --lr $lr --num-epochs 1 --seed 42 --outputs-dir trained_models/
done