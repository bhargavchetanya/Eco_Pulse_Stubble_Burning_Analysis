CREATE DATABASE IF NOT EXISTS AqiAnalysisDB;
USE AqiAnalysisDB;

CREATE TABLE Training_Data (
    record_id INT AUTO_INCREMENT PRIMARY KEY,
    date DATE,
    punjab_fire_count INT,
    wind_speed_kmph FLOAT,
    wind_dir_deg INT,
    temp_min_c FLOAT,
    delhi_aqi INT,
    aqi_category VARCHAR(50),
    dominant_reason VARCHAR(50)
);

CREATE TABLE Prediction_Log (
    pred_id INT AUTO_INCREMENT PRIMARY KEY,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    input_fire_count INT,
    input_wind_speed FLOAT,
    predicted_aqi FLOAT,
    is_severe_risk BOOLEAN,
    root_cause VARCHAR(50)
);