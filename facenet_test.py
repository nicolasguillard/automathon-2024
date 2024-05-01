import os
from facenet_pytorch import MTCNN
#import cv2
import torchvision.io as io
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

# Create face detector
mtcnn = MTCNN(select_largest=False, device='cuda')

# Load a single image and display
dataset_dir = "/raid/datasets/hackathon2024"
video_path = os.path.join(dataset_dir, "experimental_dataset", "rneucagbvg.mp4")
print(video_path)

video, audio, info = io.read_video(video_path, pts_unit='sec')
video = video.permute(0, 3, 1, 2)
length = video.shape[0]

#v_cap = cv2.VideoCapture('agqphdxmwt.mp4')
#success, frame = v_cap.read()
#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img = video[0]
frame = img.detach().numpy()

#plt.figure(figsize=(12, 8))
#plt.imshow(frame)
#plt.axis('off')

# Detect face
face = mtcnn(frame)
face.shape