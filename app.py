from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/model2.pkl", "rb") as f:
    imputer = pickle.load(f)

vehicle_types = ['Bike', 'Go Mini', 'Go Sedan', 'Premier Sedan', 'Uber XL', 'eBike']
pickup_locations = ['Badarpur', 'Barakhamba Road', 'Dwarka Sector 21', 'Khandsa',
                    'Madipur', 'Mehrauli', 'Other', 'Pataudi Chowk', 'Pragati Maidan', 'Saket']
drop_locations = ['Basai Dhankot', 'Cyber Hub', 'Kalkaji', 'Kashmere Gate ISBT', 
                  'Lajpat Nagar', 'Lok Kalyan Marg', 'Narsinghpur', 'Nehru Place', 'Other', 'Udyog Vihar']

@app.route('/')
def home():
    return render_template('index.html', vehicle_types=vehicle_types,
                           pickup_locations=pickup_locations,
                           drop_locations=drop_locations)

@app.route('/predict', methods=['POST'])
def predict():
    avg_ctat = float(request.form['avg_ctat'])
    hour = int(request.form['hour'])
    day = int(request.form['day'])
    month = int(request.form['month'])
    weekday = int(request.form['weekday'])
    is_weekend = int(request.form['is_weekend'])
    customer_total_bookings = int(request.form['customer_total_bookings'])

    vehicle = request.form['vehicle']
    pickup = request.form['pickup']
    drop = request.form['drop']

    vehicle_oh = [1 if v == vehicle else 0 for v in vehicle_types]
    pickup_oh = [1 if p == pickup else 0 for p in pickup_locations]
    drop_oh = [1 if d == drop else 0 for d in drop_locations]

    input_vector = [avg_ctat, hour, day, month, weekday, is_weekend] + \
                   vehicle_oh + pickup_oh + drop_oh + [customer_total_bookings]

    input_imputed = imputer.transform([input_vector])
    prediction = model.predict(input_imputed)[0]

    result = "🚨 Likely to cancel ride." if prediction == 1 else "✅ Unlikely to cancel ride."
    return render_template('index.html', prediction_text=result,
                           vehicle_types=vehicle_types,
                           pickup_locations=pickup_locations,
                           drop_locations=drop_locations)

if __name__ == '__main__':
    app.run(debug=True)
