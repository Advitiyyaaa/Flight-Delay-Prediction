import joblib
import pandas as pd

# Load model package
data = joblib.load("models/rf_model.pkl")

model = data['model']
le_airline = data['le_airline']
le_origin = data['le_origin']
feature_cols = data['feature_cols']  # ['DISTANCE', 'HOUR', 'MONTH', 'DAY_OF_WEEK', 'AIRLINE_CODE', 'ORIGIN']

def safe_transform(le, value):
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        return -1  # unseen category

def predict_delay(input_dict):
    # ✅ Map app keys → exact training column names
    row = {
        "HOUR":         input_dict["dep_hour"],
        "MONTH":        input_dict["month"],
        "DAY_OF_WEEK":  input_dict["day_of_week"],
        "DISTANCE":     input_dict["distance"],
        "AIRLINE_CODE": safe_transform(le_airline, input_dict["airline"]),
        "ORIGIN":       safe_transform(le_origin,  input_dict["origin"]),
    }

    df = pd.DataFrame([row])[feature_cols]  # enforce correct column order

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return pred, prob