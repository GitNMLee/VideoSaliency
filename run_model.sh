#!/bin/bash

### Some available options for GPUs are:
### geforce_rtx_2080_ti
### geforce_rtx_3090
### quadro_gp100
### rtx_a4000

sbatch --gres=gpu:rtx_a4000:1 --mem-per-cpu=32G deepgazeiie_model.sh