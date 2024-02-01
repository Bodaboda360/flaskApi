from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = load_model('bodaboda360_model.h5')
num_classes = 4
confidence_threshold = 0.8

def preprocess_image(image_file):
    # Resize the image to 224x224 pixels
    img = Image.open(image_file.stream)
    img = img.resize((224, 224))

    # Convert the image to RGB (if it's not already)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Normalize the pixel values to be between 0 and 1
    img_array = img_array / 255.0

    # Add an extra dimension for batch size
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def convert_to_serializable(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError("Type not serializable")

@app.route('/classify', methods=['POST'])
def classify_images():
    try:
        # Get the image files from the request
        images = request.files.getlist('avatars')

        # Preprocess each image
        predictions_list = []

        for image_file in images:
            img_array = preprocess_image(image_file)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])

            # Check if the confidence is above the threshold
            if confidence >= confidence_threshold and predicted_class < num_classes:
                predictions_list.append({
                    'class': int(predicted_class),
                    'confidence': confidence,
                    'message': 'Image uploaded successfully!'
                })
            else:
                predictions_list.append({
                    'class': -1,
                    'confidence': 0,
                    'message': 'Invalid image!'
                })

        return jsonify({'predictions': predictions_list})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
