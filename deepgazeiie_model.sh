#!/bin/bash
##$ -l h_rt=0:01:00             # ask for 1 hour runtime
#SBATCH --array=1-4

VIDEONUMBER=$SLURM_ARRAY_TASK_ID

module load Python/3.8.2-GCCcore-9.3.0
module load FFmpeg/4.4.2-GCCcore-11.3.0
source /homes/nmlee/virtualenvs/VideoSalienceEnv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
python /homes/nmlee/video-salience/deepgazeiie_model.py 'Video_Game.mp4' 4 $VIDEONUMBER




