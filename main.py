import numpy as np
from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('modeldiabet.pkl','rb'))

@app.route("/")
def home():
    return "Hello World"

@app.route('/api', methods=['POST']) 
def predict():
    data = request.get_json(force=True)

    datajson = {
    'age':[int(np.array(data['age']))],
    'sex' :[int(np.array(data['sex']))],
    'bmi':[int(np.array(data['bmi']))],
    'highbp':[int(np.array(data['highbp']))],
    'highchol' : [int(np.array(data['highchol']))],
    'smoker' : [int(np.array(data['smoker']))],
    'stroke' : [int(np.array(data['stroke']))],
    'heartdisease' : [int(np.array(data['heartdisease']))],
    'physactivity' : [int(np.array(data['physactivity']))],
    'diffwalk' : [int(np.array(data['diffwalk']))]
              }

    df = pd.DataFrame(datajson)
    prediction = model.predict(df)
    output = prediction['diabetes'].values[0]
    result = {}
    result['hasil'] = str(output)
    print(result['hasil'])
    return jsonify(result) 


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)