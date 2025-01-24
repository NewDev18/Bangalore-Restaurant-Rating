from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    input_data = request.form

    online_order = 1 if input_data['online_order'] == 'yes' else 0
    book_table = 1 if input_data['book_table'] == 'yes' else 0
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
