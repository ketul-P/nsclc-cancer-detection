from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = load_model('path_to_your_model.h5')

# Define the target size for your model's input
img_size = (224, 224)

# Function to preprocess the image before feeding it to the model
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the file from the request
        file = request.files['file']

        # Save the file temporarily
        img_path = 'temp_image.jpg'
        file.save(img_path)

        # Preprocess the image
        processed_img = preprocess_image(img_path)

        # Make prediction
        prediction = model.predict(processed_img)

        # Remove the temporary image
        os.remove(img_path)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
