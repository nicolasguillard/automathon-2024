import os
from facenet_pytorch import MTCNN
import cv2
#import torchvision.io as io
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

# Create face detector
mtcnn = MTCNN(select_largest=False, post_process=False, device='cuda')

# Load a single image and display
dataset_dir = "/raid/datasets/hackathon2024"
video_path = os.path.join(dataset_dir, "experimental_dataset", "rdftmwfljq.mp4")
print("video_path", video_path)

# Load a single image and display
v_cap = cv2.VideoCapture(video_path)
success, frame = v_cap.read()
print("frame", frame)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame = Image.fromarray(frame)

#plt.figure(figsize=(12, 8))
#plt.imshow(frame)
#plt.axis('off')

# Detect face
face = mtcnn(frame)
print(face.shape)