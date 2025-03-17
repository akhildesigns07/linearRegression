import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

regmodel = pickle.load(open('reg.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

@app.route('/')
def home():
    render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict_api():
    data = request.get_json['data']
    print(data)
    print(np.array(list(data.values)).reshape(1,-1))
    new_data = scaler.transform(np.array(list(data.values)).reshape(1,-1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__=="main":
    app.run(debug=True)