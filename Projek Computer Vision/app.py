from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__, static_url_path='', static_folder='static')
CORS(app)

model = load_model('model_endangered_animals.h5')
labels = {
    0: "Bekantan",
    1: "Gajah Sumatra",
    2: "Harimau Sumatra",
    3: "Monyet Langur",
    4: "OrangUtan Kalimantan"
}

@app.route('/')
def home():
    return "API is working!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' in request.files:
            file = request.files['file']
            img_bytes = BytesIO(file.read())
            img = load_img(img_bytes, target_size=(150, 150))
            img_array = img_to_array(img)

        elif 'image' in request.form:
            image_data = request.form['image']
            img_data = base64.b64decode(image_data.split(',')[1])
            img_bytes = BytesIO(img_data)
            img = Image.open(img_bytes).convert('RGB')
            img = img.resize((150, 150))
            img_array = img_to_array(img)

        else:
            return jsonify({'error': 'No image data provided'}), 400

        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][class_index]

        if confidence < 0.8:
            result = {
                'label': "Bukan Hewan Langka",
                'confidence': f"{confidence * 100:.2f}%"
            }
        else:
            result = {
                'label': f"{labels[class_index]} Merupakan Hewan Langka",
                'confidence': f"{confidence * 100:.2f}%"
            }

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/buy')
def buy():
    return send_from_directory('static', 'buy.html')

if __name__ == "__main__":
    app.run(debug=True)
