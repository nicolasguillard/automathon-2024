import os
from facenet_pytorch import MTCNN
import cv2
#import torchvision.io as io
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import torchvision

# Create face detector
mtcnn = MTCNN(select_largest=False, post_process=False, margin= 40, device='cuda')

# Load a video
dataset_dir = "/raid/datasets/hackathon2024"
video_path = os.path.join(dataset_dir, "experimental_dataset", "rdftmwfljq.mp4")
save_tensor_path = os.path.join("/raid/home/automathon_2024/account10/", 'frames_face.pt')
save_video_path = os.path.join("/raid/home/automathon_2024/account10/", 'frames_face.mp4')

def capture_face(video_path, saved_tensor_path="", saved_video_path=""):
    v_cap = cv2.VideoCapture(video_path)

    # Loop through video
    frames = []
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(v_len)):
        print(i)
        # Load frame
        success, frame = v_cap.read()
        if not success:
            continue
            
        # Add to batch, resizing for speed
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frames.append(frame)

    frames_face = mtcnn(frames)
    # When batch is full, detect faces and reset batch list
    if saved_tensor_path:
        torch.save(frames_face, saved_tensor_path)
    if saved_video_path:
        torchvision.io.write_video(frames_face, saved_video_path)


capture_face(video_path, saved_tensor_path=save_tensor_path, saved_video_path="")

