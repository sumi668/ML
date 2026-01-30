import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

flask_app = Flask(__name__)
w, b = pickle.load(open("logistic_regression.pkl","rb"))

# Normalization parameters from training data
mean = np.array([3.5593750e+01, 7.0228125e+04])
std = np.array([1.01170011e+01, 3.46007743e+04])

@flask_app.route("/") 
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    float_feature=[float(x) for x in request.form.values()]
    features = np.array(float_feature)
    # Normalize features using training data mean/std
    features_norm = (features - mean) / std
    z = np.dot(features_norm, w) + b
    prediction = "Purchased" if (1/(1+np.exp(-z))) >= 0.5 else "Not Purchased"
    return render_template("index.html",prediction_text="The Predicted value is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)


