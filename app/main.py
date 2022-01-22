from flask import Flask, request, render_template, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import os
import predict
from PIL import Image
from itertools import product


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'


def tile(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size

    grid = product(range(0, h-h % d, d), range(0, w-w % d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out)


def clean():
    if os.path.exists('./static'):
        for f in os.listdir('./static'):
            os.remove(os.path.join('./static', f))


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        clean()

        img = request.files['image']

        filename = secure_filename(img.filename)

        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        img.save(path)

        tile(filename, './uploads', './static', 256)

        possibilities = []
        for filename in os.listdir('./static'):
            prediction = predict.predict(
                os.path.join('./static', filename))

            if prediction == 'waldo':
                possibilities.append(os.path.join('./static', filename))

        return render_template('result.html', possibilities=possibilities)

    # Homepagina
    return render_template('index.html')


app.run(port=4000, debug=True)
