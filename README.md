# 🧠 AI-Based Digital Burnout & Cognitive Load Detection System

> **A full-stack AI application that predicts burnout levels, measures cognitive load, and delivers personalised wellness recommendations using Machine Learning.**

---

## 📁 Project Structure

```
burnout_detection/
│
├── app.py                         # ← Flask web application (main entry point)
├── requirements.txt               # ← Python dependencies
├── README.md
│
├── data/
│   ├── generate_dataset.py        # ← Synthetic dataset generation script
│   └── burnout_dataset.csv        # ← Generated training data (2000 samples)
│
├── models/
│   ├── train_model.py             # ← ML pipeline: preprocess, train, evaluate
│   ├── burnout_model.pkl          # ← Trained model (auto-generated)
│   ├── scaler.pkl                 # ← Feature scaler (auto-generated)
│   └── model_meta.json            # ← Model metadata & accuracy
│
├── utils/
│   ├── __init__.py
│   └── suggestions.py             # ← Rule-based smart recommendation engine
│
└── templates/
    └── index.html                 # ← Full Flask web UI
```

---

## 🚀 How to Run

### Prerequisites
- Python 3.9+
- pip

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Generate dataset + Train model
```bash
# Generate the 2000-sample synthetic dataset
python data/generate_dataset.py

# Train & evaluate all models, save best one
python models/train_model.py
```

### Step 3 — Launch the Flask app
```bash
python app.py
```

### Step 4 — Open in browser
```
http://127.0.0.1:5000
```

> **Note:** If you skip Steps 2 & 3, the app will auto-bootstrap them on first launch.

---

## 🎯 Features

### Core Features
| Feature | Description |
|---------|-------------|
| Burnout Level Prediction | Low / Medium / High using ML classification |
| Cognitive Load Score | 0–100 numerical score from behavioural signals |
| Screen Time Input | Slider input 1–16 hours |
| Typing Speed / Behaviour | WPM, backspace rate, pause frequency |
| App Usage Type | Study / Social Media / Mixed |
| Smart Suggestions | ≥8 personalised recommendations per analysis |

### Advanced Features
| Feature | Description |
|---------|-------------|
| Typing Pattern Analysis | Backspace rate + pause frequency detection |
| Distraction Detection | App switch rate → distraction index |
| Productivity Score | Composite metric from focus/breaks/distraction |
| Daily AI Report | Structured text summary of the session |
| Break Reminder Logic | Triggered when break_frequency < 3 |

---

## 🤖 Machine Learning Details

### Dataset
- **Size:** 2,000 samples, 12 features + 2 targets
- **Generation:** Domain-expert heuristics with realistic noise
- **Features:** screen time, work hours, sleep, breaks, typing speed, backspace rate, pause frequency, app switches, usage type, focus sessions, hydration, exercise
- **Feature Engineering:** 4 derived features (work-to-break ratio, sleep deficit, distraction index, typing efficiency)

### Models Compared
| Model | CV Accuracy |
|-------|------------|
| Logistic Regression | **91.25%** ✅ |
| Gradient Boosting | 88.44% |
| Random Forest | 88.00% |
| Decision Tree | 83.31% |

### Final Performance
- **Test Accuracy: 90.50%**
- **Evaluation:** 5-Fold Stratified Cross-Validation
- **Preprocessing:** StandardScaler normalisation

### Prediction Output
```json
{
  "burnout_level": "High",
  "burnout_int": 2,
  "probabilities": {"Low": 2.1, "Medium": 18.4, "High": 79.5},
  "cognitive_load": 74.3,
  "productivity": 28.1,
  "distraction_index": 91.7,
  "model_accuracy": 0.905
}
```

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web dashboard |
| `/predict` | POST | Run burnout prediction |
| `/api/model-info` | GET | Model metadata & accuracy |
| `/api/sample-data` | GET | Pre-defined demo scenarios |

---

## 🖥️ Sample Outputs

### Scenario 1 — Healthy Student
- Screen: 5h | Sleep: 8h | Work: 4h | Breaks: 6
- **Result:** Low Burnout | Cognitive Load: 18 | Productivity: 82

### Scenario 2 — Overworked Professional
- Screen: 9h | Sleep: 6h | Work: 8h | Breaks: 2
- **Result:** Medium Burnout | Cognitive Load: 51 | Productivity: 46

### Scenario 3 — Severely Burned Out
- Screen: 14h | Sleep: 4h | Work: 11h | Breaks: 0
- **Result:** High Burnout | Cognitive Load: 87 | Productivity: 12

---

## 🔮 Future Improvements

1. **OpenCV Face Fatigue Detection** — Eye blink rate & PERCLOS metric using webcam
2. **Real-time Typing Capture** — Browser keystroke API to auto-fill typing metrics
3. **User Accounts & History** — SQLite/PostgreSQL to track trends over time
4. **Push Notifications** — Break reminders via browser Notification API
5. **Wearable Integration** — Heart rate variability from smartwatch APIs
6. **NLP Journaling** — Sentiment analysis on daily text entries
7. **Mobile App** — React Native frontend with the same Flask backend

---

## 📚 References & Real-World Relevance

- World Health Organization classified burnout as a clinical syndrome (ICD-11, 2019)
- 76% of employees experience burnout at least sometimes (Gallup, 2020)
- Digital screen overuse is the #1 reported cause of cognitive fatigue in remote workers
- This system is directly applicable to corporate wellness, student monitoring, and self-care apps

---

*Built with Python · Flask · Scikit-learn · Pandas · NumPy*
