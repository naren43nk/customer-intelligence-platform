import streamlit as st
import pandas as pd
import joblib
import random
import matplotlib.pyplot as plt
import plotly.express as px
import shap

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Customer Intelligence Dashboard", layout="wide")

# ---------- DARK MODE TOGGLE ----------
dark_mode = st.sidebar.checkbox("Dark Mode")
if dark_mode:
    st.markdown("""
        <style>
        body, .stApp { background-color: #1e1e1e; color: #dcdcdc; }
        .st-bb, .st-cd, .st-bo, .st-bn { background-color: #2c2c2c !important; color: #dcdcdc; }
        </style>
    """, unsafe_allow_html=True)

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    segments = pd.read_csv('dashboard/final_dashboard_dataset.csv')
    recs = pd.read_csv('dashboard/customer_recommendations.csv', index_col=0)
    return segments, recs

@st.cache_resource
def load_model():
    return joblib.load('model/churn_model_lightgbm.pkl')

@st.cache_resource
def get_shap_explainer(_model, background_df):
    return shap.Explainer(_model, background_df)

segments_df, recs_df = load_data()
churn_model = load_model()

# ---------- SIDEBAR NAVIGATION ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Churn Prediction", "Batch Prediction", "Recommendations", "Upload Data", "About"])

# ---------- DASHBOARD ----------
if page == "Dashboard":
    st.title("Customer Intelligence Dashboard")
    st.markdown("Gain actionable insights into customer churn, segmentation, and product recommendations.")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{segments_df['CustomerID'].nunique()}")
    col2.metric("Avg. Basket Size", f"{segments_df['AvgBasketSize'].mean():.2f}")
    col3.metric("Avg. Days Between Purchases", f"{segments_df['AvgDaysBetweenPurchases'].mean():.1f}")
    col4.metric("Churn Rate (est.)", f"{(segments_df['CancellationRate'].mean() * 100):.2f} %")

    st.subheader("Customer Segment Breakdown")
    fig2, ax2 = plt.subplots()
    segments_df['Segment'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax2)
    ax2.set_ylabel('')
    st.pyplot(fig2)

# ---------- CHURN PREDICTION ----------
elif page == "Churn Prediction":
    st.title("Churn Prediction")

    if st.button("Pick Random Customer"):
        selected_customer = random.choice(segments_df['CustomerID'].unique())
    else:
        selected_customer = st.selectbox("Select CustomerID", sorted(segments_df['CustomerID'].unique()))

    row = segments_df[segments_df['CustomerID'] == selected_customer]
    segment = row['Segment'].values[0]
    st.markdown(f"**Segment:** {segment}")

    try:
        features = row[['Frequency', 'Monetary', 'TotalProductsPurchased', 'UniqueProducts',
                        'AvgBasketSize', 'AvgDaysBetweenPurchases', 'DaysSinceFirstPurchase', 'CancellationRate']]
        prob = churn_model.predict_proba(features)[0][1]
        st.markdown(f"**Churn Probability:** `{prob:.2%}`")

        if prob > 0.8:
            st.error("Very High Risk of churn")
            if segment == "Loyal Customers":
                st.info("Note: Loyal customers can still churn if their behavior drops significantly.")
            st.markdown("💡 **Action:** Send a personalized retention offer or loyalty bonus. Follow-up with a feedback request.")
        elif prob > 0.5:
            st.warning("Moderate Risk")
            st.markdown("💡 **Action:** Target with an engagement campaign or reminder about unused benefits.")
        else:
            st.success("Low Risk of churn")
            st.markdown("✅ **Action:** Continue current engagement strategy, consider upselling opportunities.")
    except Exception as e:
        st.warning(f"Prediction failed: {e}")
        features = None

    if features is not None:
        st.subheader("Why this customer is at risk (SHAP)")
        try:
            background = segments_df[['Frequency', 'Monetary', 'TotalProductsPurchased', 'UniqueProducts',
                                      'AvgBasketSize', 'AvgDaysBetweenPurchases', 'DaysSinceFirstPurchase',
                                      'CancellationRate']].sample(100, random_state=42)
            explainer = get_shap_explainer(churn_model, background)
            shap_values = explainer(features)

            try:
                fig, ax = plt.subplots()
                shap.plots.waterfall(shap_values[0], max_display=8, show=False)
                st.pyplot(fig)
            except:
                fig_bar, ax = plt.subplots()
                shap.plots.bar(shap_values[0], show=False)
                st.pyplot(fig_bar)
        except Exception as e:
            st.warning(f"SHAP failed: {e}")

# ---------- BATCH PREDICTION ----------
elif page == "Batch Prediction":
    st.title("Batch Churn Prediction")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            input_data = df[['Frequency', 'Monetary', 'TotalProductsPurchased', 'UniqueProducts',
                             'AvgBasketSize', 'AvgDaysBetweenPurchases', 'DaysSinceFirstPurchase',
                             'CancellationRate']]
            df['Churn_Probability'] = churn_model.predict_proba(input_data)[:, 1]
            st.dataframe(df)
            st.download_button("Download Predictions", df.to_csv(index=False), file_name="churn_batch_output.csv")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------- RECOMMENDATIONS ----------
elif page == "Recommendations":
    st.title("Product Recommendations")
    selected_customer = st.selectbox("Select CustomerID", sorted(segments_df['CustomerID'].unique()), key="rec")
    if selected_customer in recs_df.index:
        items = recs_df.loc[selected_customer].dropna().tolist()
        if items:
            st.success("Recommended Products:")
            for item in items:
                st.markdown(f"- {item}")
        else:
            st.info("No new recommendations.")
    else:
        st.warning("No recommendations for this customer.")

# ---------- UPLOAD DATA ----------
elif page == "Upload Data":
    st.title("Upload Data")
    uploaded = st.file_uploader("Upload your CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df)

# ---------- ABOUT ----------
elif page == "About":
    st.title("About")
    st.markdown("""
    **Built by Narendran Mohan**  
    This dashboard predicts customer churn, explains predictions using SHAP, and recommends actions to retain high-risk customers.

    **Key Features**
    - Churn prediction with LightGBM  
    - Segment-based insights  
    - SHAP explainability with fallback  
    - Batch churn prediction + CSV export  
    - Dark mode UI toggle
    """)
