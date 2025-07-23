# ⚽ xG Football Analytics Dashboard

An interactive Streamlit application for professional expected goals (xG) analysis on StatsBomb football data. Powered by a production-grade Random Forest model, this dashboard offers dynamic xG prediction, shot quality analysis, and tactical insight with a clean, modern interface.

## 🌟 Features

- **Real-time xG Prediction**  
  Configure shot scenarios via intuitive controls, with football-aware restrictions for shot type, body part, and context.

- **Shot Comparison Tool**  
  Side-by-side analysis of two distinct shot scenarios to see which has a higher xG and why.

- **xG Heatmap & Zone Analysis**  
  Visualize the probability of scoring from every area on the pitch.

- **Model Performance Dashboard**  
  Key metrics (AUC, precision, recall), model architecture, and feature importance.

- **Data Export Hub**  
  Export sample data, xG predictions, model metadata, and PDF/CSV reports.

- **Analytics Insights**  
  Tactical explanations of shot quality zones and football intelligence integration.

---

## 🚀 Live Demo

> *Deployed soon on [Streamlit Cloud]*

---

## 💻 Getting Started

1. **Clone this repository**
2. **Install dependencies**
3. **Run the app locally**

---

## 🏗️ Project Structure

xg-football-analytics/
├── app.py
├── requirements.txt
├── README.md
├── data/
│ └── processed/
│ └── sample_shots.csv
├── src/
│ └── model/
│ ├── rf_xg_model_optimized.joblib
│ ├── feature_info.json
│ └── model_metadata.json
├── notebooks/



---

## 🏆 Model Performance

| Metric       | Value   | Description                               |
|--------------|---------|-------------------------------------------|
| **AUC Score**    | 0.7918  | Professional-grade discriminatory power   |
| **Precision**    | 0.2262  | Calibration of predicted probabilities    |
| **Recall**       | 0.8131  | High true-goal detection rate             |
| **F1 Score**     | 0.3528  | Balance of precision and recall           |

---

## 💡 How It Works

- **Engine**: Random Forest classifier, hyperparameter-tuned via grid search.  
- **Data**: StatsBomb Premier League 2015/16 open data.  
- **Features**: 22 engineered variables (spatial, contextual, categorical).  
- **Logic**: Football-specific restrictions for penalties/free kicks and single-selection situational context.

---

## 🎨 UI Highlights

- Glassmorphism cards, modern gradients, and smooth animations.  
- Responsive layout for desktop and mobile.  
- Informative tooltips, icons, and contextual warnings.

---

## 📦 Export & Share

- **CSV/JSON**: Sample data, predictions, metadata.  
- **Reports**: Download detailed text/PDF summaries.  
- **Live Link**: Shareable Streamlit Cloud URL.

---

## 🤝 Contributing

Pull requests welcome! Open issues for feature requests or bug reports.

---

## 📄 License

MIT License – free for educational and research use.

---

*Built with ⚽ football passion and AI expertise!*

