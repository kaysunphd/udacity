import os
from dogclassifierapp import app
from flask import render_template, send_from_directory, request, redirect, url_for, abort, jsonify
from werkzeug.utils import secure_filename
from scripts.predict_dog_breed import predict_breed


# based on https://blog.miguelgrinberg.com/post/handling-file-uploads-with-flask
UPLOAD_PATH = 'uploads'
app.config['UPLOAD_PATH'] = UPLOAD_PATH
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg']


class Prediction:
    def __init__(self, message, image):
        self.message = message
        self.image = image


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/prediction', methods=['POST'])
def predict():
    file = request.files['file']
    filename = secure_filename(file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            abort(400)
            return redirect(url_for('index'))

        image_path = os.path.join(app.config['UPLOAD_PATH'], filename)
        file.save(image_path)

        message, image = predict_breed(image_path)
        predictions = Prediction(message, image)

        os.remove(image_path)

        return jsonify(predictions.__dict__)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)
