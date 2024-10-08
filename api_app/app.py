from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('iris_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    sepal_length = data['sepal_length']
    sepal_width = data['sepal_width']
    petal_length = data['petal_length']
    petal_width = data['petal_width']


    prediction = model.predict(np.array([[sepal_length, sepal_width, petal_length, petal_width]]))
    iris_class = ['Setosa', 'Versicolor', 'Virginica']
    result = iris_class[prediction[0]]

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(port=5001, debug=True)