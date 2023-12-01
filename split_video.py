import sys
import cv2
import time
import os

'''
Trims a video down into smaller videos. Returns how many frames each smaller video is.
'''
def split_video(video_file, num_videos):
    # List of smaller videos directories
    video_list = []

    # Open the large video file for reading
    input_video = cv2.VideoCapture(video_file)

    # Get video properties (width, height, frames per second, etc.)
    frame_width = int(input_video.get(3))  # Width
    frame_height = int(input_video.get(4))  # Height
    fps = int(input_video.get(5))  # Frames per second
    total_frames = int(input_video.get(7))  # Total number of frames

    # Specify the output folder for the smaller videos
    output_folder = f'smaller_videos_{num_videos}chunks'

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Calculate the number of frames per smaller video
    frames_per_video = total_frames // num_videos

    # Loop through and create smaller videos
    for i in range(num_videos):

        video_path = os.path.join(output_folder, f'smaller_video_{i + 1}.mp4')
        # Create a VideoWriter object for the smaller video
        output_video = cv2.VideoWriter(video_path, 
                                       cv2.VideoWriter_fourcc(*'mp4v'), 
                                       fps, 
                                       (frame_width, frame_height))
        
        video_list.append(video_path)

        # Write frames to the smaller video
        for j in range(frames_per_video):
            ret, frame = input_video.read()
            if ret:
                output_video.write(frame)
            else:
                break

        # Release the smaller video file
        output_video.release()

    # Release the large video file
    input_video.release()

    # Close all OpenCV windows (if any)
    cv2.destroyAllWindows()

def main():
    # First argument is the video file
    video_file = sys.argv[1]

    # Second argument is the number of videos to split the main video file
    num_vid = int(sys.argv[2])

    start_time = time.time()
    split_video(video_file, num_vid)
    print("Splitting video took %s seconds" % ((time.time()-start_time)))

main()