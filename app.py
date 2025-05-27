import streamlit as st
import pandas as pd
import joblib
import random
import matplotlib.pyplot as plt
import plotly.express as px

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Customer Intelligence Dashboard", layout="wide")

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    segments = pd.read_csv('dashboard/final_dashboard_dataset.csv')
    recs = pd.read_csv('dashboard/customer_recommendations.csv', index_col=0)
    return segments, recs

@st.cache_resource
def load_model():
    return joblib.load('model/churn_model_lightgbm.pkl')

segments_df, recs_df = load_data()
churn_model = load_model()

# ---------- SIDEBAR NAVIGATION ----------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Dashboard", "📉 Churn Prediction", "📦 Recommendations", "📁 Upload Data", "ℹ️ About"])

# ---------- HEADER ----------
st.title("🧠 Customer Intelligence Dashboard")
st.markdown("Gain actionable insights into customer churn, segmentation, and product recommendations.")

# ---------- KPI METRICS ----------
if page == "🏠 Dashboard":
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{segments_df['CustomerID'].nunique()}")
    col2.metric("Avg. Basket Size", f"{segments_df['AvgBasketSize'].mean():.2f}")
    col3.metric("Avg. Days Between Purchases", f"{segments_df['AvgDaysBetweenPurchases'].mean():.1f}")
    col4.metric("Churn Rate (est.)", f"{(segments_df['Cancelled'].mean() * 100):.2f} %")

    # Segment Distribution Pie Chart
    st.subheader("📊 Customer Segment Breakdown")
    fig2, ax2 = plt.subplots()
    segments_df['Segment'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax2)
    ax2.set_ylabel('')
    st.pyplot(fig2)

# ---------- CHURN PREDICTION ----------
elif page == "📉 Churn Prediction":
    st.subheader("⚠️ Predict Churn for a Customer")

    if st.button("🎲 Pick Random Customer"):
        selected_customer = random.choice(segments_df['CustomerID'].unique())
    else:
        selected_customer = st.selectbox("Select CustomerID", sorted(segments_df['CustomerID'].unique()))

    segment_row = segments_df[segments_df['CustomerID'] == selected_customer]
    st.markdown(f"**Segment:** {segment_row['Segment'].values[0]}")

    try:
        churn_features = segment_row[['Frequency', 'Monetary', 'TotalProductsPurchased',
                                      'UniqueProducts', 'AvgBasketSize', 'AvgDaysBetweenPurchases',
                                      'DaysSinceFirstPurchase', 'CancellationRate']]
        prob = churn_model.predict_proba(churn_features)[0][1]
        st.write(f"🔮 **Churn Probability:** `{prob:.2%}`")

        if prob > 0.8:
            st.error("🔴 Very High Risk of churn")
        elif prob > 0.5:
            st.warning("🟠 Moderate Risk")
        else:
            st.success("🟢 Low Risk of churn")

    except Exception:
        st.warning("Churn prediction not available for this customer.")
        import shap

# Load explainer only once
@st.cache_resource
def get_shap_explainer(model, background_df):
    return shap.Explainer(model, background_df)

st.subheader("🧠 Why this Customer is at Risk? (SHAP)")

try:
    background = segments_df[['Frequency', 'Monetary', 'TotalProductsPurchased',
                              'UniqueProducts', 'AvgBasketSize', 'AvgDaysBetweenPurchases',
                              'DaysSinceFirstPurchase', 'CancellationRate']].sample(100, random_state=42)
    
    explainer = get_shap_explainer(churn_model, background)
    shap_values = explainer(churn_features)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(shap.plots.waterfall(shap_values[0]))
except Exception as e:
    st.warning("⚠️ SHAP explanation not available for this customer.")


    # 📉 Histogram of Churn Probabilities
    st.subheader("Churn Probability Distribution (All Customers)")
    try:
        churn_features_all = segments_df[['Frequency', 'Monetary', 'TotalProductsPurchased',
                                          'UniqueProducts', 'AvgBasketSize', 'AvgDaysBetweenPurchases',
                                          'DaysSinceFirstPurchase', 'CancellationRate']]
        all_probs = churn_model.predict_proba(churn_features_all)[:, 1]
        fig = px.histogram(all_probs, nbins=30, title="Churn Probability Distribution", labels={'value': 'Churn Probability'})
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning("Could not generate churn distribution chart.")

# ---------- RECOMMENDATIONS ----------
elif page == "📦 Recommendations":
    st.subheader("🎯 Product Recommendations")

    selected_customer = st.selectbox("Select CustomerID", sorted(segments_df['CustomerID'].unique()), key="rec")

    if selected_customer in recs_df.index:
        recommended_items = recs_df.loc[selected_customer].dropna().tolist()
        if recommended_items:
            st.success("🧾 Recommended Products:")
            for item in recommended_items:
                st.markdown(f"- {item}")
        else:
            st.info("No new product recommendations.")
    else:
        st.warning("No recommendations found for this customer.")

# ---------- UPLOAD DATA ----------
elif page == "📁 Upload Data":
    st.subheader("📁 Upload Your Own Customer CSV")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        st.dataframe(user_df)

# ---------- ABOUT ----------
elif page == "ℹ️ About":
    st.subheader("About This Dashboard")
    st.markdown("""
    **Built by Narendran Mohan**  
    This dashboard is designed for customer churn prediction, segment exploration, and personalized product recommendations using real purchase behavior.

    📍 Data Source: Internal simulated customer data  
    📚 Model: LightGBM  
    📦 ML Tools: Pandas, Scikit-learn, SHAP, Plotly, Streamlit  
    """)

