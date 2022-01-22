from fastai import *
from fastai.vision import *
from fastbook import load_learner


def predict(image):
    model = load_learner('../notebook/export.pkl')
    prediction = model.predict(image)
    print(str(prediction[2][1] * 100) + '%')
    if prediction[2][1] > 0.95:
        return prediction[0]
    else:
        print("DARN")
