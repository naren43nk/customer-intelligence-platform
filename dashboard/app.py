import streamlit as st
import pandas as pd
import joblib
import random
import matplotlib.pyplot as plt

# ---- Load Data ----
@st.cache_data
def load_data():
    segments = pd.read_csv('../data/final_dashboard_dataset.csv')
    recs = pd.read_csv('../data/customer_recommendations.csv', index_col=0)
    return segments, recs

@st.cache_resource
def load_model():
    return joblib.load('../model/churn_model_lightgbm.pkl')

segments_df, recs_df = load_data()
churn_model = load_model()

# ---- Sidebar: Summary ----
st.sidebar.title("📊 Overview")
st.sidebar.markdown(f"**Total Customers:** {segments_df['CustomerID'].nunique()}")

segment_counts = segments_df['Segment'].value_counts()
st.sidebar.markdown("**Segment Distribution:**")
for seg, count in segment_counts.items():
    st.sidebar.markdown(f"- {seg}: {count}")

# ---- Main Title ----
st.title("🧠 Customer Intelligence Dashboard")

# ---- 🎲 Random Picker ----
if st.button("🎲 Pick Random Customer"):
    selected_customer = random.choice(segments_df['CustomerID'].unique())
else:
    # Fallback to first customer
    selected_customer = st.selectbox("Select CustomerID", sorted(segments_df['CustomerID'].unique()))

# ---- Retrieve Customer Row ----
segment_row = segments_df[segments_df['CustomerID'] == selected_customer]

# ---- 🎯 Segment ----
st.subheader("🎯 Customer Segment")
segment = segment_row['Segment'].values[0] if 'Segment' in segment_row else "Not available"
st.write(segment)

# ---- ⚠️ Churn Prediction ----
st.subheader("⚠️ Churn Probability")
try:
    churn_features = segment_row[['Frequency', 'Monetary', 'TotalProductsPurchased',
                                  'UniqueProducts', 'AvgBasketSize', 'AvgDaysBetweenPurchases',
                                  'DaysSinceFirstPurchase', 'CancellationRate']]
    prob = churn_model.predict_proba(churn_features)[0][1]
    st.write(f"{prob:.2%} chance of churn")

    if prob > 0.8:
        st.error("🔴 Very High Risk of churn")
    elif prob > 0.5:
        st.warning("🟠 Moderate Risk")
    else:
        st.success("🟢 Low Risk of churn")

except Exception as e:
    st.warning("⚠️ Churn prediction not available for this customer.")

# ---- 📦 Product Recommendations ----
st.subheader("📦 Product Recommendations")
if selected_customer in recs_df.index:
    recommended_items = recs_df.loc[selected_customer].dropna().tolist()
    if recommended_items:
        for item in recommended_items:
            st.markdown(f"- {item}")
    else:
        st.info("No new product recommendations.")
else:
    st.warning("No recommendations found for this customer.")

# ---- 📉 Churn Histogram ----
st.subheader("📉 Churn Probability Distribution")
try:
    churn_features_all = segments_df[['Frequency', 'Monetary', 'TotalProductsPurchased',
                                      'UniqueProducts', 'AvgBasketSize', 'AvgDaysBetweenPurchases',
                                      'DaysSinceFirstPurchase', 'CancellationRate']]
    all_probs = churn_model.predict_proba(churn_features_all)[:, 1]

    fig, ax = plt.subplots()
    ax.hist(all_probs, bins=20, color='skyblue', edgecolor='black')
    ax.set_title("Churn Probability Histogram")
    ax.set_xlabel("Churn Probability")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)
except Exception:
    st.warning("Churn distribution plot not available.")

# ---- 📊 Segment Distribution Chart ----
st.subheader("📊 Customer Segment Breakdown")
fig2, ax2 = plt.subplots()
segment_counts.plot.pie(autopct='%1.1f%%', startangle=90, ax=ax2)
ax2.set_ylabel('')
st.pyplot(fig2)

# ---- 🔄 Reset ----
if st.button("🔄 Reset"):
    st.experimental_rerun()
