# Video Saliency (DeepGazeIIE) Parallelization
The goal of this project is to run DeepGazeIIE in parallel.

The data from DeepGazeIIE is then overlayed as a heatmap on top of the original video for the output.

This series of scripts accomplishes the following tasks:
1. Trim large video into smaller subvideos (this allows for parallelization of DeepGazeIIE)
2. Run the model on a subvideo and save numpy arrays to file directory.
3. Overlay the output onto original video as a heatmap.

Example videos are in a Google Drive folder [here](https://drive.google.com/drive/folders/1jEEcrtoYTHKP-WeqKzJPLCtIYD_sAwWg?usp=sharing).

# Configuration of project
This project is meant to run on K-State's BEOCAT. If running on a different system, the driver scripts will be useless.

To configure the project to your specifications, there are a few things that must be done first.

1. Setup a virtual environment for Python
2. Configure scripts

The following sections show these configurations in more detail.

### Setup a virtual environment

In order for the project to run correctly, you need to setup a virtual environment.

To get started, follow the steps [here](https://support.beocat.ksu.edu/Docs/Installed_software#Python).

Use the following for your virtual environment:
```
module load Python/3.8.2-GCCcore-9.3.0
module load FFmpeg/4.4.2-GCCcore-11.3.0
```

Once you have created a virtual environment, install the following packages (in this order):
```
pip install matplotlib==3.3.0 opencv-python psutil scikit-video

pip install scipy==1.7.0 certifi==2020.6.20 future==0.18.2 numpy==1.19.1 Pillow==7.2.0

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```
Now your virtual environment is ready to be used!

### Configure scripts

To use your virtual environment, configure the following files:
* split_video.sh
* deepgazeiie_model.sh
* overlay_video.sh

In each of these files, change the following line to where you created your virtual environment:
```
source VIRTUAL_ENV_DIRECTORY_HERE/bin/activate

Example:
source /homes/nmlee/virtualenvs/VideoSalienceEnv/bin/activate
```

# Driver scripts
Driver scripts should be ran for main execution of the project. No other files need to be ran directly.

### File: run_split_videos.sh

The driver script for splitting a large video into smaller videos.

To configure, use split_video.sh.

The parameters for configuration are as follows:
1. The file directory of the video file to be split up
2. The number of subvideos to be created

### File: run_model.sh

The driver script for running the DeepGazeIIE model.

The GPU field can be changed to the following types, dependent upon availablity:

```
geforce_rtx_2080_ti
geforce_rtx_3090
quadro_gp100
rtx_a4000
```

Currently, using more than one GPU slows down compute time due to the DeepGazeIIE model not having a sufficient batch size.

This script parallelizes the model by running a Slurm Array Job. This can be seen at the top of the deepgazeiie_model.sh script.
```
#SBATCH --array=1-4
```
The `$SLURM_ARRAY_TASK_ID` is used for the third parameter (described below)

To configure, use deepgazeiie_model.sh

The parameters for configuration are as follows:
1. The file directory of the original video file
2. The number of subvideos that were created
3. The subvideo to run the model on

### File: run_heatmap_overlay.sh

The driver script for overlaying the original video with a heatmap.

To configure, use overlay_video.sh

The parameter for configuration are as follows:
1. The file directory of the original video file
