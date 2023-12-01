## load in the libraries we need
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import multiprocessing as mp
from multiprocessing import Pool
import time
import sys

HEATMAP_IMAGES = []
NUM_PROCESSES = 4

# Set the output video filename
HEATMAP_OUTPUT_VIDEO_FILENAME = "heatmap_video.mp4"

# Set the video frame size (adjust as needed)
#frame_size = (heatmap.shape[1], heatmap.shape[0])

# Initialize the video writer
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
OUT = cv2.VideoWriter(HEATMAP_OUTPUT_VIDEO_FILENAME, cv2.VideoWriter_fourcc(*'mp4v'), 
                                   25, (1280, 722))
"""# Output video filenames
TRIMMED_VIDEO_FILENAME = 'trimmed_video.mp4'
OUTPUT_VIDEO_FILENAME = 'output_video.mp4'

ORIGINAL_VIDEO_FILENAME = 'Video_Game.mp4'

# Open the original video
ORIGINAL_VIDEO = cv2.VideoCapture(ORIGINAL_VIDEO_FILENAME)

# Get the original video's frame width, height, and frame rate
FRAME_WIDTH = int(ORIGINAL_VIDEO.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(ORIGINAL_VIDEO.get(cv2.CAP_PROP_FRAME_HEIGHT))
FRAME_RATE = int(ORIGINAL_VIDEO.get(cv2.CAP_PROP_FPS))

# Create a VideoWriter object for the trimmed video
TRIMMED_VIDEO = cv2.VideoWriter(TRIMMED_VIDEO_FILENAME, FOURCC, FRAME_RATE, (FRAME_WIDTH, FRAME_HEIGHT))

# Create a VideoWriter object for the final output video
OUTPUT_VIDEO = cv2.VideoWriter(OUTPUT_VIDEO_FILENAME, FOURCC, FRAME_RATE, (FRAME_WIDTH, FRAME_HEIGHT))"""


def get_file_list(input_dir, output_dir):
    # Set the path to the directory containing the saved .npy files
    input_directory = input_dir
    # Create the output directory, to store the heatmap images
    output_directory = output_dir
    os.makedirs(output_directory, exist_ok=True)

    # Get a list of all files in the input directory
    file_list = os.listdir(input_directory)
    return file_list

def load_frames(file_list, input_directory, output_directory):
    # Iterate through each file in the directory
    for filename in file_list:
        if filename.endswith('.npy'):
            # Load the NumPy array from the file
            filepath = os.path.join(input_directory, filename)
            prediction_array = np.load(filepath)
            prediction_array = np.squeeze(prediction_array)

            # Create a heatmap for the loaded array
            # first, normalize the array so all values are between 0 and 1
            prediction_array_norm = (prediction_array - prediction_array.min()) / (prediction_array.max() - prediction_array.min())

            # then, make the heatmap
            print(filepath)

            # Save the heatmap as an image
            heatmap_filename = os.path.join(output_directory, f"heatmap_{filename}.png")
            plt.imsave(heatmap_filename, prediction_array_norm, cmap='jet')  # Use an appropriate colormap
            # Store the filename for later use
            HEATMAP_IMAGES.append(heatmap_filename)


## Take all images and merge them into a video

def merge_images(output_directory):
    # Specify the directory containing the heatmap images
    heatmap_images_directory = output_directory

    #Get a list of all heatmap image files in the directory
    heatmap_images = [os.path.join(heatmap_images_directory, filename) for filename in os.listdir(heatmap_images_directory) if filename.endswith('.png')]

    # Sort the heatmap images by filename (assuming filenames are numbered sequentially)
    list_of_images = sorted(heatmap_images, key=lambda x: int(os.path.basename(x).split('_prediction_')[1].split('.npy.')[0]))

    # Loop through the sorted heatmap images and add them to the video
    for heatmap_filename in list_of_images:
        #print(heatmap_filename)
        heatmap_image = cv2.imread(heatmap_filename)
        if heatmap_image is not None:
            #print(heatmap_image.shape[1])
            #print(heatmap_image.shape[0])
            heatmap_image = cv2.resize(heatmap_image,(1280, 722))
            #cv2.imshow('Heatmap Image', heatmap_image)
            #cv2.waitKey(0)  # Wait for a key press (0 means wait indefinitely)
            #cv2.destroyAllWindows()  # Close the OpenCV window
            
            OUT.write(heatmap_image)
        else:
            print(f"Warning: Unable to read image from {heatmap_filename}")
            
    # Release the video writer
    OUT.release()
    # Close all OpenCV windows (if any)
    cv2.destroyAllWindows()


def overlay(original_video_filename):
    # Input video filenames
    heatmap_video_filename = 'heatmap_video.mp4'

    ## First we transform the original video so that it matches
    ## the number of frames and size of the heatmap video 
    ## (turning Video_Game.mp4 into trimmed_video.mp4)

    # Output video filenames
    trimmed_video_filename = 'trimmed_video.mp4'
    output_video_filename = 'output_video.mp4'

    # Open the heatmap video to get its frame count
    heatmap_video = cv2.VideoCapture(heatmap_video_filename)
    heatmap_frame_count = int(heatmap_video.get(cv2.CAP_PROP_FRAME_COUNT))
    heatmap_video.release()

    # Open the original video
    original_video = cv2.VideoCapture(original_video_filename)

    # Get the original video's frame width, height, and frame rate
    frame_width = int(original_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(original_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(original_video.get(cv2.CAP_PROP_FPS))

    # Create a VideoWriter object for the trimmed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    trimmed_video = cv2.VideoWriter(trimmed_video_filename, fourcc, frame_rate, (frame_width, frame_height))

    # Process frames and write to trimmed video
    for i in range(heatmap_frame_count):
        ret, frame = original_video.read()
        if not ret:
            break  # Break the loop if we reach the end of the original video
        trimmed_video.write(frame)

    # Release video objects
    original_video.release()
    trimmed_video.release()

    ## Then we open the trimmed video & the heatmap video up, 
    ## and overlay them on top of one another.

    # Open the trimmed video and heatmap video again for overlay
    trimmed_video = cv2.VideoCapture(trimmed_video_filename)
    heatmap_video = cv2.VideoCapture(heatmap_video_filename)

    # Create a VideoWriter object for the final output video
    output_video = cv2.VideoWriter(output_video_filename, fourcc, frame_rate, (frame_width, frame_height))

    # Overlay heatmap on the trimmed video with alpha
    alpha = 0.4

    for i in range(heatmap_frame_count):
        ret1, trimmed_frame = trimmed_video.read()
        ret2, heatmap_frame = heatmap_video.read()
        
        if not (ret1 and ret2):
            break
        
        # Resize the heatmap frame to match the dimensions of the trimmed frame
        heatmap_frame = cv2.resize(heatmap_frame, (frame_width, frame_height))
        
        # Overlay the heatmap frame with alpha blending
        overlay = cv2.addWeighted(trimmed_frame, 1 - alpha, heatmap_frame, alpha, 0)
        
        # Write the overlayed frame to the output video
        output_video.write(overlay)

    # Release video objects
    trimmed_video.release()
    heatmap_video.release()
    output_video.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()


def main():
    # Video filename
    original_video_filename = sys.argv[1]

    input_directory = 'heatmap_frames'
    output_directory = 'heatmap_images'

    file_list = get_file_list(input_directory, output_directory)
    print(len(file_list))
    start_time = time.time()

    chunk = int(len(file_list) / NUM_PROCESSES)
    Processes = []
    for i in range(NUM_PROCESSES):
        start = chunk * i
        end = chunk * (i + 1)
        #print("Start: ", start, "End: ", end, "Chunk: ", chunk, "NUum Processes: ", NUM_PROCESSES)
        process = mp.Process(target=load_frames, args=(file_list[start:end], input_directory, output_directory,))
        process.start()
        Processes.append(process)
    
    #Wait for all threads to finish
    for process in Processes:
        process.join()

    print("Loading frames from the frames directory took %s seconds" % ((time.time()-start_time)))

    start_time = time.time()
    merge_images(output_directory)
    #processes = [mp.process(target=load_frames, args=(file_list)) for x in range(NUM_PROCESSES)]
    print("Taking all images and merge them into a video took %s seconds" % ((time.time()-start_time)))
    

    start_time = time.time()
    overlay(original_video_filename)
    #processes = [mp.process(target=load_frames, args=(file_list)) for x in range(NUM_PROCESSES)]
    print("Overlaying the heatmap onto the original video took %s seconds" % ((time.time()-start_time)))

main()