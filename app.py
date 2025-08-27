import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn

# Define your ResNet18 model

NUM_CLASSES = 10  
model = models.resnet18(pretrained=True)

# Freeze all layers except the last FC layer
for param in model.parameters():
    param.requires_grad = False

# Replace FC layer (must match training)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, NUM_CLASSES)
)

# Load your trained weights
model.load_state_dict(torch.load("trashNet_model.pth", map_location='cpu'))
model.eval()  # set to evaluation mode

# Preprocessing

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

#  Class labels

class_names = [
    "battery", "biological", "cardboard", "clothes", "glass",
    "metal", "paper", "plastic", "shoes", "trash"
]

# Map to reusable / non-reusable
reusable_classes = ["cardboard", "clothes", "glass", "metal", "paper", "plastic", "shoes"]


st.title("Trash Classification App")
st.write("Upload a photo or take a picture with your camera")

uploaded_file = st.file_uploader("Upload a photo", type=["jpg","png","jpeg"])
camera_photo = st.camera_input("Or take a photo")

image = None
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
elif camera_photo:
    image = Image.open(camera_photo).convert('RGB')

if image:
    st.image(image, caption="Input Image", use_container_width=True)
    input_tensor = transform(image).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
        class_label = class_names[pred.item()]
        reusable_label = "Reusable" if class_label in reusable_classes else "Non-Reusable"

    # Display results
    st.success(f"Predicted class: **{class_label}**")
    st.info(f"Category: **{reusable_label}**")
