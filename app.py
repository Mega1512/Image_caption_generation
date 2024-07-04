from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input, ResNet50
import tensorflow as tf
import pickle

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the model and tokenizer once
caption_model = tf.keras.models.load_model('models/image_captionres_model.h5')

with open('models/res_wordtoix.pkl', 'rb') as f:
    wordtoix = pickle.load(f)

with open('models/res_ixtoword.pkl', 'rb') as f:
    ixtoword = pickle.load(f)

with open('models/res_max_length.pkl', 'rb') as f:
    max_length = pickle.load(f)

base_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
encoding_model = tf.keras.models.Model(inputs=base_resnet.input, outputs=base_resnet.output)

def preprocess_img(img_path):
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def encode(image_path):
    try:
        image = preprocess_img(image_path)
        vec = encoding_model.predict(image)
        vec = np.reshape(vec, (vec.shape[1]))
        return vec
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None


def generate_caption(model, photo, max_length, wordtoix, ixtoword):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword.get(yhat, '')
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            photo = encode(file_path)
            photo = photo.reshape((1, 2048))
            caption = generate_caption(caption_model, photo, max_length, wordtoix, ixtoword)

            return render_template('index.html', caption=caption, image_url=file_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)