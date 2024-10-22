from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
model = joblib.load('svm_mnist_model.pkl')

def preprocess_image(image):
    image = image.resize((28, 28)).convert('L')
    image = np.array(image).reshape(1, -1) / 255.0
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = data['image']
    image = Image.open(io.BytesIO(base64.b64decode(image_data.split(',')[1])))
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
