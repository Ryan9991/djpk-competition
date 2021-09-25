from flask import Flask, request, url_for, redirect, render_template
import pandas as pd
import numpy as np
import pickle
from pycaret.classification import *
from sklearn import *

app = Flask(__name__, template_folder="templates", static_url_path="/static", static_folder="C:/djpk/static")

loaded_model = pickle.load(open("lr_model_realisasi.sav", 'rb'))
loaded_scaler = pickle.load(open("scaler_realisasi.sav", 'rb'))


@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def prediksi():
    input_data = [i for i in request.form.values()]
    input_final = np.array(input_data)
    input_final = input_final.astype(np.float64)
    data_scale = loaded_scaler.transform(input_final.reshape(1, -1))
    prediction = loaded_model.predict(data_scale)
    if prediction[0] == 0.0:
        return render_template('prediksi.html', prediksi = "realisasi tidak memenuhi target 100%")
    else:
        return render_template('prediksi.html', prediksi = "realisasi memenuhi target 100%")

if __name__ == '__main__':
    app.run(host = 'localhost', port = 8000, debug = True)

