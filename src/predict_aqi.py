import os
import joblib
import numpy as np
import mysql.connector
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "all_models_manual.pkl")
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
CSV_PATH = os.path.join(DATA_DIR, "delhi_aqi_data.csv")

class MyStandardScaler:
    def transform(self, X):
        return (X - self.mean) / (self.std + 1e-8)

class MyLabelEncoder:
    def inverse_transform(self, y_indices):
        return [self.inverse_classes[i] for i in y_indices]

class MyLinearRegression:
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

class MyLogisticRegression:
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def predict_proba(self, X):
        return self._sigmoid(np.dot(X, self.weights) + self.bias)

class MyNeuralNetwork:
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def predict(self, X):
        a1 = self._sigmoid(np.dot(X, self.W1) + self.b1)
        return np.argmax(self._sigmoid(np.dot(a1, self.W2) + self.b2), axis=1)

db_config = {
    'user': 'root',
    'password': 'password',
    'host': 'localhost',
    'database': 'AqiAnalysisDB'
}

def get_aqi_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"

def make_prediction():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    data = joblib.load(MODEL_PATH)
    lin_reg = data['linear']
    log_reg = data['logistic']
    nn_model = data['neural_net']
    scaler = data['scaler']
    le = data['encoder']

    print("\n--- MANUAL AI SYSTEM ---")
    try:
        fire = float(input("Punjab Fire Count: "))
        wind = float(input("Wind Speed (km/h): "))
        direction = float(input("Wind Direction (deg): "))
        temp = float(input("Temperature (C): "))
    except ValueError:
        print("Invalid input")
        return

    X = scaler.transform(np.array([[fire, wind, direction, temp]]))
    aqi = lin_reg.predict(X)[0]
    prob = log_reg.predict_proba(X)[0]
    severe = bool(prob > 0.5)
    reason = le.inverse_transform([nn_model.predict(X)[0]])[0]

    print("\n==============================")
    print(f"Predicted AQI: {aqi:.2f}")
    print(f"High Risk Probability: {prob * 100:.1f}%")
    print(f"Dominant Cause: {reason}")
    print("==============================")

    if save_to_log(fire, wind, aqi, severe, reason):
        if input("\nAdd to training data? (y/n): ").lower() == 'y':
            add_to_training(fire, wind, direction, temp, aqi, reason)
    else:
        print("\nDatabase offline")

def save_to_log(fire, wind, aqi, risk, cause):
    try:
        conn = mysql.connector.connect(**db_config)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO Prediction_Log
            (input_fire_count, input_wind_speed, predicted_aqi, is_severe_risk, root_cause)
            VALUES (%s,%s,%s,%s,%s)
        """, (fire, wind, float(aqi), risk, cause))
        conn.commit()
        cur.callproc('Generate_Alert', [float(aqi), cause])
        for r in cur.stored_results():
            print(f"\nDATABASE ALERT: {r.fetchone()[0]}")
        return True
    except mysql.connector.Error:
        return False

def add_to_training(fire, wind, direction, temp, aqi, cause):
    today = datetime.date.today()
    category = get_aqi_category(aqi)

    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(CSV_PATH, 'a') as f:
            f.write(f"\n{today},{int(fire)},{wind},{direction},{temp},{int(aqi)},{category},{cause}")
        print("Added to CSV")
    except Exception:
        pass

    try:
        conn = mysql.connector.connect(**db_config)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO Training_Data
            (date, punjab_fire_count, wind_speed_kmph, wind_dir_deg,
             temp_min_c, delhi_aqi, aqi_category, dominant_reason)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """, (today, fire, wind, direction, temp, aqi, category, cause))
        conn.commit()
        print("Added to database")
    except mysql.connector.Error:
        pass

if __name__ == "__main__":
    make_prediction()

