from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the Breast Cancer dataset
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target)

# Create a pipeline that combines Standard Scaler and Logistic Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('clf', LogisticRegression(max_iter=10000))  # Increase max_iter for convergence
])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Fit the pipeline with the training data
pipeline.fit(X_train, y_train)

@app.route('/')
def home():
    # Pass feature names to the template
    return render_template('index.html', feature_names=cancer.feature_names)

@app.route('/predict', methods=['POST'])
def predict_cancer_route():
    # Get data from form
    data = request.form.to_dict()
    
    # Convert form data to a list of feature values
    try:
        features = [float(data[feature]) for feature in cancer.feature_names]
    except KeyError as e:
        return f"Missing feature value: {e}", 400
    
    # Make prediction
    prediction = pipeline.predict([features])[0]
    result = "Positive" if prediction == 1 else "Negative"
    
    return render_template('index.html', feature_names=cancer.feature_names, prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
