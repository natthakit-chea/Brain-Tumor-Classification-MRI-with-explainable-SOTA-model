import streamlit as st
from rembg import remove
from PIL import Image
from io import BytesIO
import base64
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import time
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst
import numpy as np
import re
import torch.nn.functional as F

st.set_page_config(layout="wide", page_title="Brain tumer classification")

st.write("## Brain tumer classification")
st.write(
    "To Detect and Classify Brain Tumor using, CNN and TL; as an asset of Deep Learning and to examine the tumor position(segmentation)"
)
st.sidebar.write("## Upload and download :gear:")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50_model = torch.load('saved/resnet50_full_model.pth')
resnet50_model = resnet50_model.to(device)
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

def prediction(model, image, class_names):
    model.eval()
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.to(device)  # Move input tensor to the correct device
    with torch.no_grad():
        output = model(input_tensor)
    probs = F.softmax(output, dim=1)

    # Get the predicted class and its confidence score
    _, predicted_idx = torch.max(probs, 1)
    predicted_class = class_names[predicted_idx.item()]
    confidence_score = probs[0, predicted_idx].item()

    return predicted_idx.item(), confidence_score

def grad_cam(model,image, target):
    target_layers = [model.layer4[-1]]
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    targets = [ClassifierOutputTarget(target)]
    rgb_img = np.array(image.resize((224, 224))) / 255.0
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets,aug_smooth=True, eigen_smooth=True)[0, :]
        visualization = show_cam_on_image(rgb_img.astype(np.float32), grayscale_cam, use_rgb=True)
        # print(type(visualization))
        PIL_image = Image.fromarray(np.uint8(visualization)).convert('RGB')

        PIL_image = Image.fromarray(visualization.astype('uint8'), 'RGB')
    return PIL_image.resize((512, 512))

def convert_image(img):
    # Ensure the image is in RGB mode
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im



def fix_image(upload):
    image = Image.open(upload)
    col1.write("Original Image :camera:")
    col1.image(image.resize((512, 512)))
    # print(image.size)

    # fixed = remove(image)
    traget_idx,confident =  prediction(resnet50_model,image,class_names)
    percent = confident*100
    col2.write(f'Predicted class: **{class_names[traget_idx]}** with confidence score: **{percent:.2f}%**')
    if traget_idx != 2:
        pre_image = grad_cam(resnet50_model,image,traget_idx)
    else:
        pre_image = image.resize((512, 512))
    col2.image(pre_image)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download Predicted image", convert_image(pre_image), "pre_image.png", "image/png")


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        fix_image(upload=my_upload)
else:
    fix_image("/home/natthakit/304proj/dataset/val/meningioma/Te-me_0012.jpg")

# streamlit run app.py --server.port 6008