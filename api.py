from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms
from models.mdl_mnist_202520 import ConvolutionalNeuralNetwork
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)  # ✅ Allow all origins by default

# Load model
model = ConvolutionalNeuralNetwork()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

# Define image transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

from PIL import Image, ImageOps

def preprocess_image(file):
    # Load image
    image = Image.open(file)

    # If image has alpha channel, composite on white background to remove transparency
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        background = Image.new('RGBA', image.size, (255, 255, 255, 255))  # white background
        background.paste(image, mask=image.split()[-1])  # paste using alpha channel as mask
        image = background.convert('RGB')  # convert to RGB without alpha

    # Convert to grayscale
    image = image.convert('L')

    # Invert colors: make background white and digit black
    image = ImageOps.invert(image)

    # Resize to 28x28 (model input size)
    image = image.resize((28, 28))

    # plt.imshow(image, cmap='gray')
    # plt.show()
    # Convert to tensor and normalize (if you want, otherwise just ToTensor)
    image = transform(image).unsqueeze(0)

    return image



def predict_digit(image_tensor):
    """Run prediction on image tensor using the loaded model."""
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.item()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        image_tensor = preprocess_image(file)
        print("Image shape:", image_tensor.shape)
        prediction = predict_digit(image_tensor)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
