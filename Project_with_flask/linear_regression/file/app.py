import numpy as np
from flask import Flask,request,render_template
import pickle
import os


flask_app = Flask(__name__, template_folder="templates", static_folder="static")
script_dir = os.path.dirname(os.path.abspath(__file__))
pkl_path = os.path.join(script_dir, "linear_regression.pkl")

with open(pkl_path, "rb") as f:
    w, b = pickle.load(f)


mean= np.array([69.438, 4.983625, 6.52375, 4.5745])
std=np.array([17.3581798, 2.58691068, 1.69548988, 2.85774382])

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/performance", methods=["POST"])
def performance():
    float_features=[float(x) for x in request.form.values()]
    features = np.array(float_features)
    features_norm= (features-mean)/std
    performance = float(np.dot(features_norm,w)+b)
    print("Performance:", performance)
    return render_template("index.html", performance="StudentPerformance: {:.2f}".format(performance))

if __name__=="__main__":
    flask_app.run(debug=True)
