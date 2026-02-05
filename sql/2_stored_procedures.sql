USE AqiAnalysisDB;

DELIMITER //

CREATE PROCEDURE Generate_Alert(IN aqi_val FLOAT, IN cause VARCHAR(50))
BEGIN
    DECLARE alert_msg VARCHAR(255);
    
    IF aqi_val > 400 THEN
        SET alert_msg = CONCAT('CRITICAL ALERT: Severe Smog likely due to ', cause);
    ELSEIF aqi_val > 300 THEN
        SET alert_msg = CONCAT('WARNING: Poor Air Quality due to ', cause);
    ELSE
        SET alert_msg = 'INFO: Air Quality is manageable.';
    END IF;
    
    SELECT alert_msg AS System_Notification;
END //

DELIMITER ;