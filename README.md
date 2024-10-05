# Image Classification with PyTorch (Multiple Models)

This project is an interactive web application for **image classification** using multiple **pre-trained deep learning models** in PyTorch. Users can upload an image, choose from various models, and classify the image in real-time!

## 🚀 Live Web App
Check out the live web app here:  
[Image Classification Web App](https://your-deployed-app-link.com)

## 🌟 Features
- **Upload Image**: Upload any image (JPG, PNG) and classify it using a selected pre-trained model.
- **Choose Model**: Select from three popular models: **ResNet50**, **EfficientNet**, or **MobileNetV2**.
- **Real-time Classification**: Get instant predictions with the top predicted class displayed.
- **Explore ImageNet Classes**: View the full list of **ImageNet** classes.
- **Read Model Papers**: Learn more about each model by accessing the original research papers.

## 🛠️ Tech Stack
- **Streamlit**: For building the interactive web interface.
- **PyTorch**: For loading pre-trained models and performing classification.
- **torchvision**: For accessing pre-trained models and image transformations.
- **Pillow**: For handling uploaded image files.

## 📦 How to Run Locally

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/ImageClassificationPyTorchApp.git
    ```
   
2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the app**:
    ```bash
    streamlit run app.py
    ```

4. **Open the app** in your browser at:
    ```bash
    http://localhost:8501
    ```

## 🖼️ Example Use Case
Upload an image of an **animal**, **vehicle**, or **object**, and the app will classify it using the selected model from ResNet, EfficientNet, or MobileNet.

## 📄 Read More About Each Model
- **ResNet (Residual Networks)**: [Paper](https://arxiv.org/abs/1512.03385)
- **EfficientNet**: [Paper](https://arxiv.org/abs/1905.11946)
- **MobileNetV2**: [Paper](https://arxiv.org/abs/1801.04381)

## 💡 Future Work
- **Fine-tuning**: Add functionality to fine-tune the models on custom datasets.
- **Additional Models**: Add more models like Inception, DenseNet, etc.
- **Batch Inference**: Allow classification of multiple images at once.

## License
This project is licensed under the MIT License.
