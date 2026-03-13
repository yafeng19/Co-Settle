import random
import decord
import numpy as np
import torch
import os
from torchvision import transforms

from PIL import Image


# Set decord to use torch
decord.bridge.set_bridge('torch')

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor()        
])

def sample_one_frame(video_path, repeated_sampling=1):
    """
    Sample a pair of frames from a video with a random gap between them.
    """

    frame_files = sorted(os.listdir(video_path))
    n_frames = len(frame_files)
    frame_idx = random.randint(0, n_frames - 1)

    with Image.open(os.path.join(video_path, frame_files[frame_idx])) as img:
        frame = TRANSFORM(img.convert("RGB"))

    return frame    # torch.Size([3, 224, 224])



def sample_two_frames(video_path, clip_gap=0.15):
    """
    Sample a pair of frames from a video with a random gap between them.
    """

    frame_files = sorted(os.listdir(video_path))
    n_frames = len(frame_files)
    frame_gap = int(clip_gap*n_frames)

    if n_frames <= 2 or frame_gap <= 1:
        frame1_idx = random.randint(0, n_frames - 1)
        frame2_idx = random.randint(0, n_frames - 1)
    else:
        # Set gap and sample 3 indices
        lower_bound = n_frames - frame_gap

        frame1_idx = np.random.randint(1, lower_bound)   # np.random.randint not contain upper_bound
        frame2_idx = frame1_idx + frame_gap

    idx_lst = [frame1_idx, frame2_idx]
    frame_lst = []

    for idx in idx_lst:
        with Image.open(os.path.join(video_path, frame_files[idx])) as img:
            frame = TRANSFORM(img.convert("RGB"))
            frame_lst.append(frame)

    return tuple(frame_lst)



def sample_two_frames_tuple(video_path, clip_gap=0.15):
    """
    Sample a pair of frames from a video with a random gap between them.
    """

    frame_files = sorted(os.listdir(video_path))
    n_frames = len(frame_files)
    frame_gap = int(clip_gap*n_frames)

    if n_frames <= 4 or frame_gap < 1:
        frame1_idx = random.randint(0, n_frames - 1)
        frame2_idx = frame1_idx
        frame3_idx = random.randint(0, n_frames - 1)
        frame4_idx = frame3_idx
    else:
        # Set gap and sample 2 indices
        lower_bound = n_frames - frame_gap
        # group 1
        frame2_idx = np.random.randint(1, lower_bound)   # np.random.randint not contain upper_bound
        frame1_idx = max(frame2_idx - 1, 0)
        # group 2
        frame4_idx = frame2_idx + frame_gap
        frame3_idx = max(frame4_idx - 1, 0)

    idx_lst = [frame1_idx, frame2_idx, frame3_idx, frame4_idx]
    frame_lst = []

    for idx in idx_lst:
        with Image.open(os.path.join(video_path, frame_files[idx])) as img:
            frame = TRANSFORM(img.convert("RGB"))
            frame_lst.append(frame)

    return tuple(frame_lst)

