import os
import joblib
import numpy as np
import mysql.connector
import datetime
import random

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

def get_aqi_category(aqi_value):
    if aqi_value <= 50: return "Good"
    elif aqi_value <= 100: return "Satisfactory"
    elif aqi_value <= 200: return "Moderate"
    elif aqi_value <= 300: return "Poor"
    elif aqi_value <= 400: return "Very Poor"
    else: return "Severe"

def make_prediction():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    brain_data = joblib.load(MODEL_PATH)
    aqi_model = brain_data['linear']
    risk_classifier = brain_data['logistic']
    cause_predictor = brain_data['neural_net']
    data_scaler = brain_data['scaler']
    label_mapping = brain_data['encoder']

    print("\n--- ECO-PULSE PREDICTION SYSTEM ---")
    try:
        farm_fires = float(input("Enter Punjab Farm Fire Count: "))
        velocity = float(input("Enter Wind Speed (km/h): "))
        bearing = float(input("Enter Wind Direction (degrees): "))
        temperature = float(input("Enter Minimum Temperature (C): "))
    except ValueError:
        print("Error: Please enter numerical values.")
        return

    input_features = data_scaler.transform(np.array([[farm_fires, velocity, bearing, temperature]]))
    raw_aqi_result = aqi_model.predict(input_features)[0]

    if raw_aqi_result < 0:
        final_aqi = random.randint(1, 50)
    else:
        final_aqi = raw_aqi_result

    danger_probability = risk_classifier.predict_proba(input_features)[0]
    is_critical = bool(danger_probability > 0.5)
    primary_reason = label_mapping.inverse_transform([cause_predictor.predict(input_features)[0]])[0]

    print("\n==============================")
    print(f"Final Predicted AQI: {final_aqi:.2f}")
    print(f"Probability of Severe Air: {danger_probability * 100:.1f}%")
    print(f"Predicted Dominant Cause: {primary_reason}")
    print("==============================")

    if save_to_log(farm_fires, velocity, final_aqi, is_critical, primary_reason):
        if input("\nWould you like to store this in training data? (y/n): ").lower() == 'y':
            add_to_training(farm_fires, velocity, bearing, temperature, final_aqi, primary_reason)
    else:
        print("\nNote: Database offline.")

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
            print(f"\nDATABASE NOTIFICATION: {r.fetchone()[0]}")
        return True
    except mysql.connector.Error:
        return False

def add_to_training(fire, wind, direction, temp, aqi, cause):
    timestamp = datetime.date.today()
    air_quality_label = get_aqi_category(aqi)

    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(CSV_PATH, 'a') as f:
            f.write(f"\n{timestamp},{int(fire)},{wind},{direction},{temp},{int(aqi)},{air_quality_label},{cause}")
        print("✅ Locally saved.")
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
        """, (timestamp, fire, wind, direction, temp, aqi, air_quality_label, cause))
        conn.commit()
        print("✅ Pushed to SQL.")
    except mysql.connector.Error:
        pass

if __name__ == "__main__":
    make_prediction()
