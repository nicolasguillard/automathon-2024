import os
import json
import csv

import cv2
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import torchvision.io as io
import torchvision.transforms as transforms

from facenet_pytorch import MTCNN

from tqdm import tqdm


mtcnn = MTCNN(image_size=200, select_largest=False, post_process=False, margin=40, device='cuda')

dataset_dir = "/raid/datasets/hackathon2024"
root_dir = os.path.expanduser("~/automathon-2024")
save_dir = "/raid/home/automathon_2024/account10/data"

save_tensor = True
save_video = True
margin = 25
for dataset_choice in ["train", "test", "experimental"]:
        if  dataset_choice == "train":
            root_dir = os.path.join(dataset_dir, "train_dataset")
            save_path = os.path.join(save_dir, "train_dataset")
        elif dataset_choice == "test":
            root_dir = os.path.join(dataset_dir, "test_dataset")
            save_path = os.path.join(save_dir, "test_dataset")
        elif dataset_choice == "experimental":
            root_dir = os.path.join(dataset_dir, "experimental_dataset")
            save_path = os.path.join(save_dir, "experimental_dataset")
        else:
            raise ValueError("choice must be 'train', 'test' or 'experimental'")
        
        for file in os.listdir(root_dir):
            if file.endswith('.mp4'):
                video_path = os.path.join(root_dir, file)
                print(video_path)

                video, audio, info = io.read_video(video_path, pts_unit='sec')

                v_cap = cv2.VideoCapture(video_path)
                v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(v_cap.get(cv2.CAP_PROP_FPS))

                frames = []
                boxes = []
                faces = []
                min_l = 98765
                
                for _ in tqdm(range(v_len)):
                    # Load frame
                    success, frame = v_cap.read()
                    if not success:
                        continue
                    
                    # Change chanels
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame.astype(np.uint8)
                    
                    # Process boxes
                    img = Image.fromarray(frame)
                    batch_boxes, _, _ = mtcnn.detect(img, landmarks=True) # (x1, y1, x2, y2)
                    box = batch_boxes[0]
                    box = np.array(box, dtype=int)
                    
                    c_w, c_h = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
                    w = box[2] - box[0]
                    h = box[3] - box[1]
                    l = max(w, h)
                    box_square = [
                        c_w - l//2 - margin,
                        c_h - l//2 - margin,
                        c_w + l//2 + margin,
                        c_h + l//2 + margin
                    ]
                    
                    #bs_x1
                    if box_square[0] < 0:
                        box_square[0] = 0
                        box_square[2] -= box_square[0]
                    #bs_y1
                    if box_square[1] < 0:
                        box_square[1] = 0
                        box_square[3] -= box_square[1]
                    #bs_x2
                    if box_square[2] >= frame.shape[0]:
                        box_square[2] = frame.shape[0] - 1
                        box_square[0] += frame.shape[0] - box_square[2]
                    #bs_y2
                    if box_square[3] >= frame.shape[1]:
                        box_square[3] = frame.shape[1] - 1
                        box_square[1] += frame.shape[1] - box_square[3]
                    
                    face = frame[box_square[1]:box_square[3], box_square[0]:box_square[2], :]
                    faces.append(torch.tensor(face))
                    min_l = min(l, min_l)

                # Rescaling each frame
                print(min_l, min_l.item())
                transform = transforms.Resize(min_l.item())
                for i, face in tqdm(enumerate(faces)):
                    faces[i] = transform(face)

                frames_face = torch.cat(faces)
                    
                # Save files
                saved_tensor_path = os.path.join(save_path, file.replace(".mp4", "_face.pt"))
                print(f"\tsaving {saved_tensor_path}")
                if save_tensor:
                    torch.save(frames_face, saved_tensor_path)

                saved_video_path = os.path.join(save_path, file.replace(".", "_face."))
                print(f"\tsaving {saved_video_path}")
                if save_video:
                    torchvision.io.write_video(frames_face, saved_video_path, fps=fps)
