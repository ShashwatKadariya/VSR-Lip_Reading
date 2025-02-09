import cv2
import tensorflow as tf
from typing import List
from model.utils import char_to_num
import os

def load_video(path: str) -> List[float]:
    """
    Load and preprocess a video from the specified path.

    Parameters:
        path (str): The path to the video file.

    Returns:
        List[float]: A list of preprocessed frames from the video."""


    # Open the video file
    cap = cv2.VideoCapture(path)

    # Initialize an empty list to store frames
    frames = []

    # Loop through all frames in the video
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        # Read a frame from the video
        ret, frame = cap.read()

        # Convert the frame to grayscale
        frame = tf.image.rgb_to_grayscale(frame)

        # Crop the frame to the region of interest i.e. lip region
        frames.append(frame[190:236, 80:220, :])

    # Release the video capture object
    cap.release()

    # Calculate mean and standard deviation for normalization
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))

    # Normalize frames and cast to float32
    return tf.cast((frames - mean), tf.float32) / std

def load_alignments(path:str) -> List[str]:
    """
    Load alignments from a text file at the specified path.

    Parameters:
        path (str): The path to the text file containing alignments.

    Returns:
        List[str]: A list of phonetic tokens extracted from the alignments file.
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    # Initialize an empty list to store phonetic tokens
    tokens = []

    # Iterate through each line in the file
    for line in lines:
        line = line.split()
        # Check if the token is not 'sil' (silence)
        if line[2] != 'sil':
            tokens = [*tokens,' ',line[2]]

    # Convert phonetic tokens to numerical indices using char_to_num layer, and exclude the first space token
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def load_data(path: str, video_path="../data/s1/", alignment_path="../data/alignments/s1/"):
    """
    Load data from the specified path

    Parameters:
        path (str): The path to the data file.
        video_path (str): The path to the video file.
        alignment_path (str): The path to the alignment file.


    Returns:
        Tuple[List[float], List[str]]: A tuple containing the loaded video frames and alignments.
    """
    # Convert path from bytes to string
    path = bytes.decode(path.numpy())
    # Extract file name from the path
    file_name = path.split('/')[-1].split('.')[0]
    # Construct paths for video and alignment files
    video_path = os.path.join(video_path, f'{file_name}.mpg')
    alignment_path = os.path.join(alignment_path, f'{file_name}.align')
    # Load video frames
    frames = load_video(video_path)

    # Load alignments
    alignments = load_alignments(alignment_path)

    return frames, alignments

def mappable_function(path: str) -> List[str]:
    """
    A mappable function to load data from the specified path.

    Parameters:
        path (str): The path to the data file.

    Returns:
        List[str]: A list containing the loaded video frames and alignments.
    """
    # Call load_data using tf.py_function
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))

    return result