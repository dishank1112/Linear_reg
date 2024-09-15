from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the ridge regressor model and scaler
ridge_model = pickle.load(open('./models/rid.pkl', 'rb'))
Standard_Scaler = pickle.load(open('./models/scale.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('home.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        CLASSES = float(request.form.get('CLASSES'))
        Region = float(request.form.get('Region'))

        new_data = Standard_Scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, CLASSES, Region]])
        result = ridge_model.predict(new_data)

        return render_template('home.html', results=result[0])
    else:
        render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")
