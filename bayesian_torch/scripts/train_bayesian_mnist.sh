#!/bin/bash

project_name='tighter bounds'
obj="pacm"
lr=1.0
batch_size=256
epochs=40
mode='train'
save_dir='./checkpoint/bayesian'

python examples/main_bayesian_mnist.py --project-name=$project_name \
                                        --obj=$obj \
                                        --lr=$lr \
                                        --batch-size=$batch_size \
                                        --epochs=$epochs \
                                        --mode=$mode \
                                        --save_dir=$save_dir

