from scipy.misc import face
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
import deepgaze_pytorch

import cv2
import numpy as np
import os
import time
import torch.nn as nn
from itertools import repeat
import sys

'''
Chooses the device for modeling. If CUDA is available, it will be used.
Otherwise, CPU will be used.
'''
def choose_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # Check the available GPU memory
        gpu_properties = torch.cuda.get_device_properties(device)
        gpu_memory = gpu_properties.total_memory / (1024**2)  # in megabytes
        print(f"GPU Name: {gpu_properties.name}")
        print(f"GPU Memory: {gpu_memory} MB")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU.")
        cpu_memory = psutil.virtual_memory().available / (1024**2)  # in megabytes
        print(f"CPU Memory: {cpu_memory} MB")
    return device

'''
Load the DeepGazeIIE model
'''
def load_model(device):
    return deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(device)

'''
Runs the DeepGazeIIE model on the given video file. Saves results in output_directory
'''
def run_model(video_file, model, device, starting_frame, output_directory):
    # Create a VideoCapture object and read from input file 
    video = cv2.VideoCapture(video_file) 
    
    # Check if video opened successfully 
    if (video.isOpened()== False): 
        print("Error opening video file") 

    # load precomputed centerbias log density (from MIT1003) over a 1024x1024 image
    centerbias_template = np.load('centerbias_mit1003.npy')
    
    i = starting_frame

    # To prevent from resizing each time, we hold this variable to be reused each time.
    centerbias_tensor = None

    # Read until video is completed 
    while(video.isOpened()):
        # Capture frame-by-frame 
        ret, frame = video.read() 
        if ret == True:
            image = frame

            # If this is the first frame, resize centerbias.
            if i == starting_frame:
                # rescale to match image size
                centerbias = zoom(centerbias_template, (image.shape[0]/centerbias_template.shape[0], image.shape[1]/centerbias_template.shape[1]), order=0, mode='nearest')
                # renormalize log density
                centerbias -= logsumexp(centerbias)
                # Save to variable outside loop
                centerbias_tensor = torch.tensor(np.array([centerbias])).to(torch.float32).to(device)
            
            image_tensor = torch.tensor(np.array([image.transpose(2, 0, 1)])).to(torch.float32).to(device)

            log_density_prediction = model(image_tensor, centerbias_tensor)
            torch.cuda.empty_cache()

            filename = f'{output_directory}/prediction_{i}.npy'
            np.save(filename, log_density_prediction.cpu().detach().numpy())
            print(f"Saved prediction {i} to {filename}")

            i += 1

    # Break the loop 
        else: 
            break
    
    # Release the video capture object 
    video.release() 
    
    # Close all frames 
    cv2.destroyAllWindows()

def main():
    # First argument is the video file
    video_file = sys.argv[1]

    # Second argument is the number of videos to split the main video file
    num_of_vids = int(sys.argv[2])

    # Third argument is which subvideo this is script is running on
    sub_video = int(sys.argv[3])

    input_video = cv2.VideoCapture(video_file)
    total_frames = int(input_video.get(7))  # Total number of frames
    frames_per_video = total_frames // num_of_vids

    input_video.release() 

    start_time = time.time()
    DEVICE = choose_device()
    print("Choosing device took %s seconds" % ((time.time()-start_time)))

    start_time = time.time()
    model = load_model(DEVICE)
    model = nn.DataParallel(model)
    print("Loading model took %s seconds" % ((time.time()-start_time)))

    ## set output directory
    output_directory = f'heatmap_frames'
    os.makedirs(output_directory, exist_ok=True)

    start_time = time.time()
    run_model(os.path.join(f'smaller_videos_{num_of_vids}chunks', f'smaller_video_{sub_video}.mp4'), model, DEVICE, (sub_video-1) * frames_per_video, output_directory)
    print("Running model took %s seconds" % ((time.time()-start_time)))

main()