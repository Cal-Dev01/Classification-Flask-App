from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
data_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'classification_data.csv')
model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'classification_model.pkl')
graph_path = os.path.join(app.config['UPLOAD_FOLDER'], 'classification_plot.png')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def save_data(data):
    """Save new data to the existing data file."""
    if os.path.exists(data_file_path):
        existing_data = pd.read_csv(data_file_path)
        updated_data = pd.concat([existing_data, data], ignore_index=True)
    else:
        updated_data = data
    updated_data.to_csv(data_file_path, index=False)


def train_model():
    """Train the Logistic Regression model using the data from the data file."""
    data = pd.read_csv(data_file_path)
    X = data[['feature1', 'feature2']]
    y = data['label']

    # Train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X, y)

    # Save the model
    joblib.dump(model, model_path)

    # Generate and save the graph with decision boundary
    plt.figure()
    plt.scatter(data['feature1'], data['feature2'], c=y, cmap='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Classification')
    plt.savefig(graph_path)
    plt.close()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        data = pd.read_csv(file)

        if 'feature1' in data.columns and 'feature2' in data.columns and 'label' in data.columns:
            save_data(data)
            train_model()
            return jsonify(
                {'message': 'Data uploaded and model trained successfully!', 'graph_url': url_for('get_graph')})
        else:
            return jsonify({'error': 'CSV file must contain feature1, feature2, and label columns'}), 400


@app.route('/graph')
def get_graph():
    return send_file(graph_path, mimetype='image/png')


@app.route('/predict', methods=['POST'])
def predict():
    feature1 = request.form['feature1']
    feature2 = request.form['feature2']
    data = [[float(feature1), float(feature2)]]

    if not os.path.exists(model_path):
        return jsonify({'error': 'Model not found, please upload a CSV file to train the model'}), 400

    model = joblib.load(model_path)
    prediction = model.predict(data)
    return jsonify({'prediction': int(prediction[0])})


if __name__ == '__main__':
    app.run(debug=True)
