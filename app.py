import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import requests

# Load models from torchvision
@st.cache(allow_output_mutation=True)
def load_model(model_name):
    if model_name == "ResNet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "EfficientNet":
        model = models.efficientnet_b0(pretrained=True)
    elif model_name == "MobileNetV2":
        model = models.mobilenet_v2(pretrained=True)
    
    model.eval()  # Set to evaluation mode
    return model

# Preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Predict the class of the image
def predict_image_class(image, model):
    image = preprocess_image(image)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    
    # Get class names from ImageNet
    labels_map = load_labels_map()
    predicted_class = labels_map[predicted.item()]
    return predicted_class

# Load ImageNet class labels
@st.cache(allow_output_mutation=True)
def load_labels_map():
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    labels = requests.get(url).json()
    return labels

# Title of the app
st.title("Image Classification with PyTorch (Multiple Models)")

st.markdown("""
Upload an image and classify it using one of the pre-trained deep learning models in PyTorch! 
Choose from **ResNet50**, **EfficientNet**, or **MobileNetV2** and get instant predictions.
""")

# File uploader for users to upload images
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Dropdown to select the model
model_name = st.selectbox("Choose a model", ["ResNet50", "EfficientNet", "MobileNetV2"])

# Load the selected model
model = load_model(model_name)

if uploaded_file is not None:
    # Load the uploaded image
    img = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    st.write("Classifying using **{}**...".format(model_name))
    
    # Predict the class of the image
    predicted_class = predict_image_class(img, model)
    
    # Display the predicted class
    st.write(f"**Prediction:** {predicted_class}")
