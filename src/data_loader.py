import pandas as pd
import mysql.connector

db_config = {
    'user': 'root',
    'password': 'password',
    'host': 'localhost',
    'database': 'AqiAnalysisDB'
}

def load_data_to_db():
    try:
        df = pd.read_csv('../data/delhi_aqi_data.csv')
    except FileNotFoundError:
        return

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        for index, row in df.iterrows():
            sql = """
            INSERT INTO Training_Data 
            (date, punjab_fire_count, wind_speed_kmph, wind_dir_deg, temp_min_c, delhi_aqi, aqi_category, dominant_reason)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            values = (
                row['date'],
                row['punjab_fire_count'],
                row['wind_speed_kmph'],
                row['wind_dir_deg'],
                row['temp_min_c'],
                row['delhi_aqi'],
                row['aqi_category'],
                row['dominant_reason']
            )
            cursor.execute(sql, values)
            
        conn.commit()
        
    except mysql.connector.Error as err:
        print(err)
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == "__main__":
    load_data_to_db()
