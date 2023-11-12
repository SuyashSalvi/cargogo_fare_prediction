from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the pickled model
with open('your_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        input_data = request.get_json()
        new_input = pd.DataFrame(input_data)

        # Make predictions using the loaded model
        predictions = model.predict(new_input)

        # Return the predictions as JSON
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000)
