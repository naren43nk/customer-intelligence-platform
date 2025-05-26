# 🧠 Customer Intelligence Platform

A complete data science pipeline to analyze customer behavior, predict churn, segment users, and generate personalized product recommendations — built using real-world transactional data from the **UCI Online Retail II Dataset**.

This project includes **data cleaning, ML modeling, customer segmentation, recommendations, and an interactive Streamlit dashboard**, ideal for showcasing in portfolios or business analytics demos.

---

## 📚 Notebooks Overview

### 1️⃣ Data Cleaning & EDA
- Cleaned transaction logs from the Online Retail II dataset (UCI)
- Removed cancellations, missing values, and engineered `TotalPrice`
- Basic insights into customer purchase behavior

### 2️⃣ Churn Prediction
- Labeled churn based on Recency thresholds
- Engineered features: product diversity, basket size, time gaps
- Trained **XGBoost & LightGBM** with SMOTE
- Achieved ~**99% ROC-AUC**, with SHAP explainability

### 3️⃣ RFM Segmentation
- Calculated Recency, Frequency, Monetary values
- Clustered customers using `scipy.kmeans2` (macOS-compatible)
- Visualized using PCA and labeled segments (e.g., Champions, At Risk)

### 4️⃣ Recommendation System
- User-based collaborative filtering
- Cosine similarity on customer-product matrix
- Generated top-N product suggestions per customer

---

## 💻 Streamlit Dashboard

An interactive app to explore insights:

- 🎯 View customer segments
- ⚠️ Predict churn probability (with interpretation)
- 📦 Get personalized product recommendations
- 📊 Visualize segment distribution & churn trends
- 🎲 Random customer sampling

### ▶️ To Launch Dashboard Locally

``bash
cd dashboard
streamlit run app.py


## Getting Started

### Clone this repo
git clone https://github.com/naren43nk/customer-intelligence-platform.git
cd customer-intelligence-platform

### Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

### Install dependencies
pip install -r requirements.txt

### Start exploring notebooks
jupyter notebook


### Tech Stack
Python 3.10+
Pandas, NumPy, Scikit-learn, LightGBM, XGBoost
Matplotlib, SHAP, SciPy
Streamlit
Jupyter Notebooks

### Dataset Source
UCI Online Retail II Dataset
https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

👨‍💻 Author
Narendran Mohan
MSc Data Science | University of Surrey
🔗 [https://www.linkedin.com/in/narendran-mohan-a5801a333](https://www.linkedin.com/in/narendran-mohan-a5801a333)

⭐ Highlights
* End-to-end ML lifecycle implementation
* Realistic churn mitigation and targeting
* Portfolio-ready, dashboard-powered, recruiter-friendly
* Explains business value and modeling in one clean app

### Feedback
If you like this project or found it useful, feel free to ⭐ the repo or connect on LinkedIn. Suggestions and contributions welcome!
