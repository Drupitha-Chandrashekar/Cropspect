from flask import Flask, request, jsonify
import numpy as np
from torchvision import transforms
import torch
from PIL import Image
import io
import joblib  # Assuming you need to load environmental model as well

# Initialize the Flask app
app = Flask(__name__)

# Load the image model
image_model = torch.load('image_model.pth', map_location=torch.device('cpu'))  # Replace with your actual image model file path
image_model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])


@app.route('/')
def home():
    return "Welcome to the image prediction service!"


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        print('No image field found in request.')
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    
    if image_file.filename == '':
        print('No file selected.')
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        print(f'File received: {image_file.filename}')
        image = Image.open(io.BytesIO(image_file.read()))
        image = image.convert('RGB')  # Ensure image is in RGB format
        image_tensor = transform(image).unsqueeze(0)
        
        # Predict using the model
        with torch.no_grad():
            outputs = image_model(image_tensor)
            _, preds = torch.max(outputs, 1)
            prediction = preds.item()  # Get the predicted class index

        return jsonify({'prediction': prediction})
    
    except Exception as e:
        print(f'Error: {e}')
        return jsonify({'error': str(e)}), 400




if __name__ == '__main__':
    # Run the Flask app
    app.run(port=8083)
