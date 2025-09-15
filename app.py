import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Mall Customer Segmentation", page_icon="🛍️", layout="wide")

st.title("🛍️ Mall Customer Segmentation")

@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

df = load_data()

# Sidebar
st.sidebar.header("⚙️ Settings")
k = st.sidebar.slider("Number of clusters (k)", 2, 10, 5)

# Preprocess
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans
model = KMeans(n_clusters=k, random_state=42)
clusters = model.fit_predict(X_scaled)
df['Cluster'] = clusters

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(8), height=200)

with col2:
    st.subheader(f"📈 Customer Segments (k={k})")
    fig, ax = plt.subplots(figsize=(3, 2), dpi=80)  # much smaller
    scatter = ax.scatter(
        df['Annual Income (k$)'],
        df['Spending Score (1-100)'],
        c=df['Cluster'], cmap='viridis', s=15
    )
    ax.set_xlabel("Income", fontsize=7)
    ax.set_ylabel("Score", fontsize=7)
    ax.tick_params(axis='both', labelsize=6)
    plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.01)
    st.pyplot(fig, clear_figure=True, use_container_width=False)

st.markdown("---")
st.markdown("<div style='text-align:center; color:gray'>🛍️ Mall Customer Segmentation | Streamlit</div>", unsafe_allow_html=True)
