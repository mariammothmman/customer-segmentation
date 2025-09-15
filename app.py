import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Mall Customer Segmentation", page_icon="🛍️", layout="wide")
st.title("🛍️ Mall Customer Segmentation")

@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

df = load_data()

st.sidebar.header("⚙️ Settings")
k = st.sidebar.slider("Number of clusters (k)", 2, 10, 5)

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"📈 Customer Segments (k={k})")
    fig = px.scatter(
        df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color="Cluster",
        template="plotly_white",
        width=400, height=300
    )
    fig.update_traces(marker=dict(size=6))
    st.plotly_chart(fig, use_container_width=False)

with col2:
    st.subheader("📊 Cluster Distribution")
    cluster_counts = df['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    fig2 = px.bar(
        cluster_counts,
        x="Cluster", y="Count", color="Cluster",
        template="plotly_white", width=400, height=300, text="Count"
    )
    st.plotly_chart(fig2, use_container_width=False)

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:13px'>🛍️ Mall Customer Segmentation | Streamlit</div>", 
    unsafe_allow_html=True
)
