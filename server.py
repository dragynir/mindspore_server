from flask import Flask, jsonify, request
import numpy as np
from model import Predictor

app = Flask(__name__)
predictor = Predictor()

@app.route("/predict", methods=['POST'])
def predict():

    data = request.get_json()
    params = data['params']
    x = np.array(data['arr'])
    out = predictor.predict(x)

    return jsonify({"Status": 'ok', 'out': out.tolist()}), 200

if __name__ == '__main__':
    port = 9097
    host = '0.0.0.0'
    print(f"Starting server {host}:{port}")
    app.run(debug=False, host=host, port=port, threaded=True)