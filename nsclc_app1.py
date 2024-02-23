import os
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import base64
import requests

app = Flask(__name__, template_folder='templates')
model_file = "best_model.h5"
model = load_model(model_file)

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the target size for your model's input
img_size = (224, 224)

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
    prediction = int(np.argmax(predictions))  # Assuming it's a classification task

    if prediction == 1:  # Adjust this condition based on your model output
        result = "Positive"
    else:
        result = "Negative"

    return result

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if request.form['phoneNumber'] != "":
            phonenumber = request.form['phoneNumber']
            dictToSend = {
                'phoneNumber': phonenumber,
                'recordType': 'Your_Record_Type'
            }
            # Adjust the URL and payload based on your API requirements
            res = requests.post('https://your-api-endpoint', json=dictToSend)
            dictFromServer = res.json()
            
            img_data = dictFromServer['success'][0]['file']['buffer']
            myimage = base64.b64decode(img_data)

            with open(os.path.join(app.config['UPLOAD_FOLDER'], "imageFetched.jpeg"), "wb") as fh:
                fh.write(myimage)

            predictions = make_predictions(os.path.join(app.config['UPLOAD_FOLDER'], "imageFetched.jpeg"))
            return render_template('index.html', filename="imageFetched.jpeg", message=predictions, show=True)

        elif request.files['img'] != '':
            f = request.files['img']
            filename = 'uploaded_image.jpeg'
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            predictions = make_predictions(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('index.html', filename=filename, message=predictions, show=True)

    return render_template('index.html', filename='unnamed.png')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
