import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

app = Flask(__name__, template_folder='templates')
model_file = "best_model.keras"
model = load_model(model_file)

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the target size for your model's input
img_size = (256, 256)

# Function to preprocess the image before feeding it to the model
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def make_predictions(img_path):
    '''
    Method to predict based on the image uploaded
    '''
    processed_img = preprocess_image(img_path)
    predictions = model.predict(processed_img)
    prediction = int(np.argmax(predictions))

    if prediction == 0:
        result = "Adenocarcinoma"
    elif prediction == 1:
        result = "Large Cell Carcinoma"
    elif prediction == 3:  # Corrected index for "Pneumonia"
        result = "Pneumonia"
    elif prediction == 4:  # Corrected index for "Squamous Cell Carcinoma"
        result = "Squamous Cell Carcinoma"
    else:
        result = "Normal"

    return result

@app.route('/', methods=[GET])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['img']
        filename = 'uploaded_image.jpeg'
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        predictions = make_predictions(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_template('index.html', message=predictions, show=True)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')