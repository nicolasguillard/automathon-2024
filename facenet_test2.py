import os
from facenet_pytorch import MTCNN, InceptionResnetV1

# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN(image_size=<image_size>, margin=<margin>)

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()

from PIL import Image

dataset_dir = "/raid/datasets/hackathon2024"
video_path = os.path.join(dataset_dir, "experimental_dataset", "rdftmwfljq.mp4")
img = Image.open()

# Get cropped and prewhitened image tensor
save_path = "/raid/home/automathon_2024/account10/"
img_cropped = mtcnn(img, save_path=os.path.join(save_path, "image_cropped"))

# Calculate embedding (unsqueeze to add batch dimension)
img_embedding = resnet(img_cropped.unsqueeze(0))

# Or, if using for VGGFace2 classification
resnet.classify = True
img_probs = resnet(img_cropped.unsqueeze(0))
print(img_probs)