import numpy as np
from flask import Flask,request,jsonify, render_template
import pickle


flask_app = Flask(__name__, template_folder="templates", static_folder="static")
w,b = pickle.load(open("linear_regression.pkl","rb"))

mean= np.array([69.438, 4.983625, 6.52375, 4.5745])
std=np.array([17.3581798, 2.58691068, 1.69548988, 2.85774382])

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/performance", methods=["POST"])
def predict():
    float_features=[float(x) for x in request.form.values()]
    features = np.array(float_features)
    features_norm = (features - mean) / std
    x=np.dot(features_norm, w) + b
    performance = "Good" if (1/(1+np.exp(-x))) >= 0.5 else "Bad"
    return render_template("index.html", performance="StudentPerformance: {}".format(performance))

if __name__=="__main__":
    flask_app.run(debug=True)
