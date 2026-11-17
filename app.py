'''
In this file we are going to write the backend code to load the model and give prediction
'''
import flask
from flask import Flask,render_template,request
import numpy as np
import pickle
app = Flask(__name__)
from sklearn.linear_model import LinearRegression

with open("SLR_model.plk" , "rb") as f:
    model = pickle.load(f)

@app.route("/")
def default_site():
    return render_template("index.html")

@app.route("/predict",methods = ['GET','POST'])
def prediction():
    value = [float(i) for i in request.form.values()]  #  [15.0]
    value = np.array([value])  # [ [15.0]  ]
    result = model.predict(value)[0]  # []
    return render_template("index.html" , prediction_text = result)



if __name__ == "__main__":
    app.run(debug=True)
