from flask import Flask, render_template, request
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load('churn_model.pkl')

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract user input from the form
        age = float(request.form['age'])
        gender = request.form['gender']
        location = request.form['location']
        subscription_length = float(request.form['subscription_length'])
        monthly_bill = float(request.form['monthly_bill'])
        total_usage_gb = float(request.form['total_usage_gb'])

        # Create a DataFrame with the user input
        user_input = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Location': [location],
            'Subscription_Length_Months': [subscription_length],
            'Monthly_Bill': [monthly_bill],
            'Total_Usage_GB': [total_usage_gb]
        })

        # Preprocess user input (if necessary)
        # Example: You may need to apply the same preprocessing as done for training data

        # Make predictions using the loaded model
        churn_prediction = model.predict(user_input)

        # Return the prediction to the HTML template
        return render_template('form.html', prediction=churn_prediction[0])
    except Exception as e:
        return render_template('form.html', prediction=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
