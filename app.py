import streamlit as st
import pandas as pd
from utils.predict import predict_delay

st.set_page_config(page_title="Flight Delay Predictor", layout="wide")

# -------------------------------
# Airline Mapping (Full Name → Code)
# -------------------------------

airline_map = {
    "Alaska Airlines Inc.": "AS",
    "American Airlines Inc.": "AA",
    "Delta Air Lines Inc.": "DL",
    "United Air Lines Inc.": "UA",
    "Southwest Airlines Co.": "WN",
    "JetBlue Airways": "B6",
    "SkyWest Airlines Inc.": "OO",
    "ExpressJet Airlines Inc.": "EV",
    "Frontier Airlines Inc.": "F9",
    "Spirit Air Lines": "NK",
    "Hawaiian Airlines Inc.": "HA"
}

# -------------------------------
# Load Data & Build Lookups
# -------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/flights_clean.csv")

@st.cache_data
def get_distance_map(df):
    """Fallback: median distance per origin airport"""
    return df.groupby('ORIGIN_CITY')['DISTANCE'].median().to_dict()

@st.cache_data
def get_route_distance_map(df):
    """Precise: median distance per (origin, dest) route"""
    return df.groupby(['ORIGIN_CITY', 'DEST_CITY'])['DISTANCE'].median().to_dict()

@st.cache_data
def get_valid_destinations(df, origin):
    """Only destinations reachable from selected origin"""
    return sorted(df[df['ORIGIN_CITY'] == origin]['DEST_CITY'].unique())

df = load_data()
distance_map = get_distance_map(df)
route_distance_map = get_route_distance_map(df)
airports = sorted(df['ORIGIN_CITY'].unique())

# -------------------------------
# Sidebar Inputs
# -------------------------------

st.sidebar.header("✈️ Enter Flight Details")

airline_name = st.sidebar.selectbox("Airline", list(airline_map.keys()))
airline = airline_map[airline_name]

origin = st.sidebar.selectbox("Origin Airport", airports)

# Destinations filtered to only valid routes from selected origin
valid_dests = get_valid_destinations(df, origin)
dest = st.sidebar.selectbox("Destination Airport", valid_dests)

dep_hour = st.sidebar.slider("Departure Hour", 0, 23, 12)
month = st.sidebar.slider("Month", 1, 12, 6)
day_of_week = st.sidebar.selectbox(
    "Day of Week",
    options=[0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x]
)

# Auto-lookup distance: precise route first, fallback to origin median
distance = route_distance_map.get((origin, dest), distance_map.get(origin, 500))

# -------------------------------
# Main UI
# -------------------------------

st.title("✈️ Flight Delay & Cancellation Prediction System")
st.markdown("Enter flight details from the sidebar and click **Predict**.")

# -------------------------------
# Prediction
# -------------------------------

if st.sidebar.button("Predict"):
    input_data = {
        "airline":     airline,
        "origin":      origin,
        "dep_hour":    dep_hour,
        "month":       month,
        "day_of_week": day_of_week,
        "distance":    distance,
    }

    pred, prob = predict_delay(input_data)

    st.subheader("📊 Prediction Result")
    col1, col2 = st.columns(2)

    with col1:
        if pred == 1:
            st.error("⚠️ Flight likely to be delayed")
        else:
            st.success("✅ Flight likely on time")

    with col2:
        st.metric("Delay Probability", f"{prob*100:.2f}%")

    st.caption(f"📏 Route: {origin} → {dest} | Distance: **{int(distance)} miles**")

# -------------------------------
# Dashboard
# -------------------------------

st.header("📊 Insights Dashboard")

try:
    col1, col2 = st.columns(2)
    with col1:
        st.image("images/plots/airline_delay_bar.png", caption="Delay by Airline")
    with col2:
        st.image("images/plots/airport_delay.png", caption="Delay by Airport")

    col3, col4 = st.columns(2)
    with col3:
        st.image("images/plots/time_of_day_delay.png", caption="Delay by Time of Day")
    with col4:
        st.image("images/plots/monthly_trend.png", caption="Monthly Delay Trend")

    col5, col6 = st.columns(2)
    with col5:
        st.image("images/plots/delay_histogram.png", caption="Delay Distribution")
    with col6:
        st.image("images/plots/delay_causes.png", caption="Delay Causes")

    col7, col8 = st.columns(2)
    with col7:
        st.image("images/plots/correlation_heatmap.png", caption="Feature Correlation Heatmap")
    with col8:
        st.image("images/plots/feature_importance.png", caption="Feature Importance")

    st.image("images/plots/confusion_matrix.png", caption="Confusion Matrix")

except Exception as e:
    st.warning(f"Some dashboard images could not be loaded: {e}")

# -------------------------------
# Footer
# -------------------------------

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Flight Delay ML Project")