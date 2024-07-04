from flask import Flask, request, jsonify
import joblib
from flask import render_template

app = Flask(__name__)
model_spam = joblib.load('./models/spam_classifier.pkl')
model_churn = joblib.load('./models/svc_churn.pkl')
vectoriser = joblib.load('./models/vectoriser')



def encode_inputs(data):
    encoded_data = []
    encoded_data.append(data['tenure'])
    encoded_data.append(data['MonthlyCharges'])
    encoded_data.append(data['TotalCharges'])
    encoded_data.append(data['gender'] == 'Male')
    encoded_data.append(data['SeniorCitizen'] == 'Yes')
    encoded_data.append(data['MultipleLines'] == 'No phone service')
    encoded_data.append(data['MultipleLines'] == 'Yes')
    encoded_data.append(data['Partner'] == 'Yes')
    encoded_data.append(data['Dependents'] == 'Yes')
    encoded_data.append(data['PhoneService'] == 'Yes')
    encoded_data.append(data['MultipleLines'] == 'No phone service')
    encoded_data.append(data['MultipleLines'] == 'Yes')
    encoded_data.append(data['InternetService'] == 'Fiber optic')
    encoded_data.append(data['InternetService'] == 'No')
    encoded_data.append(data['OnlineSecurity'] == 'No internet service')
    encoded_data.append(data['OnlineSecurity'] == 'Yes')
    encoded_data.append(data['OnlineBackup'] == 'No internet service')
    encoded_data.append(data['OnlineBackup'] == 'Yes')
    encoded_data.append(data['Contract'] == 'One year')
    encoded_data.append(data['Contract'] == 'Two year')
    encoded_data.append(data['StreamingMovies'] == 'No internet service')
    encoded_data.append(data['StreamingMovies'] == 'Yes')
    return encoded_data


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/churn')
def churn():
    return render_template('churn.html')



@app.route('/spam')
def spam():
    return render_template('spam.html')


    
@app.route('/churnPrediction', methods=['POST'])
def predict_churn():
    data = request.form
    encoded_data = encode_inputs(data)
    prediction = model_churn.predict([encoded_data])[0]
    return jsonify({'prediction': 'YES' if prediction else 'NO'})


@app.route('/spamPrediction', methods=['POST'])
def predict_spam():
    data = request.form
    message = data['message']
    message = vectoriser.transform([message])
    prediction = model_spam.predict(message)[0]
    return jsonify({'prediction': 'SPAM' if prediction else 'NOT SPAM'})


if __name__ == '__main__':
    app.run()