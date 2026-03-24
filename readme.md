# ✈️ Flight Delay Prediction System

A machine learning web application that predicts whether a flight will be delayed (>15 minutes departure delay) using a Random Forest Classifier, served via a Streamlit dashboard.

---

## 📁 Project Structure

```
Flight Delay Analysis/
│
├── app.py                    # Streamlit web application
├── data/
│   └── flights_clean.csv     # Cleaned flight dataset
├── models/
│   └── rf_model.pkl          # Trained Random Forest model + encoders
├── utils/
│   └── predict.py            # Prediction logic
├── images/
│   └── plots/                # Dashboard visualizations
│       ├── airline_delay_bar.png
│       ├── airport_delay.png
│       ├── confusion_matrix.png
│       ├── correlation_heatmap.png
│       ├── delay_causes.png
│       ├── delay_histogram.png
│       ├── feature_importance.png
│       ├── monthly_trend.png
│       └── time_of_day_delay.png
└── notebooks/
    ├── DataProcessing.ipynb  # Data cleaning & EDA
    ├── Train.ipynb           # Model training
    └── Test.ipynb            # Model evaluation
```

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Training Rows | 1,500,000 (subsampled) |
| Dataset Size | ~2.9 million flights |
| Accuracy | 65.29% |
| Delay Rate in Data | 17.5% |
| Target | `DELAYED` — 1 if departure delay > 15 min, else 0 |

### Features Used

| Feature | Description |
|---|---|
| `HOUR` | Departure hour (0–23) |
| `DISTANCE` | Route distance in miles (auto-looked up) |
| `AIRLINE_CODE` | Airline carrier code (label encoded) |
| `ORIGIN` | Origin airport (label encoded) |
| `MONTH` | Month of flight (1–12) |
| `DAY_OF_WEEK` | Day of week (0=Mon … 6=Sun) |

### Feature Importance

| Rank | Feature | Importance |
|---|---|---|
| 1 | HOUR | 41.7% |
| 2 | DISTANCE | 16.5% |
| 3 | AIRLINE_CODE | 12.7% |
| 4 | ORIGIN | 12.5% |
| 5 | MONTH | 10.9% |
| 6 | DAY_OF_WEEK | 5.7% |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/flight-delay-predictor.git
cd flight-delay-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
```

Create a `requirements.txt` with the above, or install manually:

```bash
pip install streamlit pandas numpy scikit-learn joblib matplotlib seaborn
```

---

## 🖥️ App Usage

1. Open the app in your browser (default: `http://localhost:8501`)
2. Fill in the sidebar inputs:
   - **Airline** — select from 11 major US carriers
   - **Origin Airport** — departure city
   - **Destination Airport** — filtered to valid routes from origin
   - **Departure Hour** — scheduled departure time
   - **Month** — month of travel
   - **Day of Week** — day of travel
3. Click **Predict**
4. View the delay prediction and probability score
5. Explore the **Insights Dashboard** below for visual analytics

> 💡 Distance is automatically looked up from the route data — no manual input needed.

---

## 📊 Dashboard Visualizations

| Chart | Description |
|---|---|
| Delay by Airline | Average delay rate per carrier |
| Delay by Airport | Top airports by delay frequency |
| Delay by Time of Day | How departure hour affects delays |
| Monthly Delay Trend | Seasonal patterns in delays |
| Delay Distribution | Histogram of delay minutes |
| Delay Causes | Breakdown by delay reason |
| Correlation Heatmap | Feature correlation matrix |
| Feature Importance | Random Forest feature importance scores |
| Confusion Matrix | Model classification performance |

---

## 🔧 Notebooks

| Notebook | Purpose |
|---|---|
| `DataProcessing.ipynb` | Raw data cleaning, feature engineering, EDA |
| `Train.ipynb` | Model training, encoding, saving `rf_model.pkl` |
| `Test.ipynb` | Model evaluation, metrics, plots |

---

## ✈️ Supported Airlines

| Airline | Code |
|---|---|
| Alaska Airlines Inc. | AS |
| American Airlines Inc. | AA |
| Delta Air Lines Inc. | DL |
| United Air Lines Inc. | UA |
| Southwest Airlines Co. | WN |
| JetBlue Airways | B6 |
| SkyWest Airlines Inc. | OO |
| ExpressJet Airlines Inc. | EV |
| Frontier Airlines Inc. | F9 |
| Spirit Air Lines | NK |
| Hawaiian Airlines Inc. | HA |

---

## ⚠️ Limitations

- Model accuracy is 65.29% — predictions are probabilistic, not guaranteed
- Only trained on US domestic flights
- Destination is used solely for distance lookup; it does not directly affect the prediction
- Weather, air traffic control, and real-time factors are not included

---

## 📄 License

This project is for educational purposes.

---

Built with ❤️ using Streamlit | Scikit-learn | Python