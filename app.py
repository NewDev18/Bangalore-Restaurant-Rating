from flask import Flask, request, render_template
import pickle
import numpy as np
import json
import pandas as pd
import string

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    input_data = request.form
    
    #with open(r'fetures_name.json','r') as file:
    #   features_name = json.load(file)

    online_order = int(input_data['online_order'])
    book_table = int(input_data['book_table'])
    votes = int(input_data['votes'])
    location = int(input_data['location'])
    rest_type = int(input_data['rest_type'])
    cuisines = int(input_data['cuisines'])
    cost = float(input_data['cost'])
    # store input independent features into array

    arr = np.array([online_order, book_table, votes, location,
                    rest_type, cuisines, cost], ndmin=2)
    print(arr)
    # loading Model

    with open(r'model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Predicting the results

    result = model.predict(arr)
    rounded_result = np.round(result, decimals=1)
    print(rounded_result)

    return render_template('index.html', prediction = rounded_result)

if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0')
