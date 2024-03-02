import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import PIL
import requests
import base64
from flask_cors import CORS  

app = Flask(__name__, template_folder='templates')
CORS(app, resources={r"/*": {"origins": "*"}})


model_file = "best_model.h5"
model = load_model(model_file)

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the target size for your model's input
img_size = (224, 224)

# Function to preprocess the image before feeding it to the model
def preprocessImage(img_path):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array



def makePredictions(img_path):
    '''
    Method to predict based on the image uploaded
    '''
    processed_img = preprocessImage(img_path)
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

@app.route('/', methods=['GET', 'POST'])
def home():
    if(request.method=='POST'):
        # Check if the 'img' field is present in the request
        if request.files['img'] != "":
            img_file = request.files['img']
            # Save the image to the static folder
            filename = img_file.filename
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img_file.save(image_path)

            # Perform predictions
            predictions = makePredictions(os.path.join(app.config['UPLOAD_FOLDER'],filename))

            render_template('home.html',filename=img_file.filename,message=predictions,show=True)
            return jsonify({'message': predictions, 'show': True})

        render_template('home.html',filename="unnamed.png",message='Image not uploaded',show=True)
        return jsonify({'Error': 'Image not uploaded'}), 400
      
    return jsonify({'Error': 'Invalid Request'}), 400  
    
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
