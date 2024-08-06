from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd

application = Flask(__name__)
app=application
#import ridge regressior and StandardScaler
import pickle

# Correct file paths
ridge= pickle.load(open('C:/Users/anil.nagamunthala/flask/linear_rigression_ridge.pkl', 'rb'))
scaler = pickle.load(open('C:/Users/anil.nagamunthala/flask/linear_rigression_scaler.pkl', 'rb'))


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoints():
    if request.method == "POST":
        
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))  # Assuming Classes is an integer
        Region = int(request.form.get('Region'))
        new_data=scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result=ridge.predict(new_data)
        return render_template('home.html',results=result[0])
     
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")

