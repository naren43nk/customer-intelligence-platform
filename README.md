# 🧠 Customer Intelligence Platform

A complete data science pipeline to analyze customer behavior, predict churn, segment users, and generate personalized product recommendations — built using real-world transactional data from the **UCI Online Retail II Dataset**.

This project demonstrates **data cleaning, machine learning, customer segmentation, recommendations, and an interactive Streamlit dashboard**, ideal for portfolios, business use cases, or data science showcases.

---

## 📁 Project Structure

customer-intelligence-platform/
│
├── dashboard/
│ ├── app.py ← Streamlit dashboard code
│ ├── final_dashboard_dataset.csv
│ └── customer_recommendations.csv
│
├── model/
│ └── churn_model_lightgbm.pkl ← Pre-trained churn model
│
├── notebooks/
│ ├── 1_data_cleaning_eda.ipynb
│ ├── 2_churn_prediction.ipynb
│ ├── 3_rfm_segmentation.ipynb
│ └── 4_recommendation_system.ipynb
│
├── requirements.txt
└── README.md


---

## 📚 Notebook Overview

### 1️⃣ Data Cleaning & EDA
- Cleaned UCI transaction logs (removal of cancellations, NA rows)
- Feature engineered `TotalPrice`, filtered UK customers
- Visualized basic purchasing behavior

### 2️⃣ Churn Prediction
- Defined churn via Recency thresholding
- Feature set includes diversity, basket metrics, time gaps
- Trained **XGBoost** and **LightGBM** with **SMOTE**
- Achieved ~**99% ROC-AUC**, with **SHAP** model explainability

### 3️⃣ RFM Segmentation
- Computed Recency, Frequency, and Monetary scores
- Applied `scipy.kmeans2` clustering (Mac-friendly)
- Reduced dimensions with PCA and labeled segments:
  - Champions, Loyal, Potential Loyalist, At Risk, etc.

### 4️⃣ Recommendation System
- User-based collaborative filtering
- Cosine similarity on customer-product sparse matrix
- Generated **top-N personalized suggestions** per customer

---

## 💻 Streamlit Dashboard

### Features:
- **Segment Explorer** – View distribution across customer types
- **Churn Prediction** – Estimate churn probability per user
- **SHAP Explainability** – Understand key churn drivers
- **Batch Prediction** – Upload CSV for mass churn analysis
- **Recommendations** – See personalized product suggestions
- **Dark Mode UI** – Clean toggle for improved readability

---
## Quickstart

## ▶️ Launch Dashboard Locally

cd customer-intelligence-platform/dashboard
streamlit run app.py

# 1. Clone the repo
git clone https://github.com/naren43nk/customer-intelligence-platform.git
cd customer-intelligence-platform

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit dashboard
cd dashboard
streamlit run app.py

## Tech Stack

Language: Python 3.10+

Libraries:

Pandas, NumPy, Scikit-learn, SciPy

LightGBM, XGBoost

SHAP, Matplotlib, Plotly

Streamlit

Dev Tools: Jupyter Notebook, Git

## Dataset
Source: UCI Online Retail II Dataset
Details: Historical transactions from a UK-based online retailer, includes invoices, dates, and product codes

## Author
Narendran Mohan
MSc Data Science, University of Surrey
🔗 LinkedIn

## Highlights
End-to-end implementation of ML lifecycle

Real-world business context: churn mitigation + retention

Visual explainability using SHAP

Portfolio-ready, recruiter-friendly Streamlit UI

## Feedback
⭐ If you found this helpful, star the repo and connect on LinkedIn!
Pull requests, suggestions, and feedback are welcome.

Let me know if you want me to:
- Add a **LICENSE.md**
- Auto-generate a fresh `requirements.txt`
- Set up GitHub Actions or deployment instructions via Streamlit Cloud
