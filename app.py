import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="SmartCart AI Dashboard",
    page_icon="🛒",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #111827);
    color: white;
}
section[data-testid="stSidebar"] {
    background-color: #111827;
}
.metric-card {
    background: rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("🛒 SmartCart AI Customer Intelligence Dashboard")
st.markdown("### AI-powered customer segmentation for intelligent business insights")

uploaded_file = st.file_uploader(
    "Upload SmartCart Customer Dataset",
    type=["csv"]
)

if uploaded_file:

    try:
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        df.columns = df.columns.str.strip()
    except Exception as e:
        st.error(f"Error parsing the CSV file: {e}")
        st.stop()

    key_columns = ["Income", "Year_Birth", "MntWines", "MntFruits", "MntMeatProducts",
                   "MntFishProducts", "MntSweetProducts", "MntGoldProds", "Recency",
                   "NumWebPurchases", "NumStorePurchases", "NumCatalogPurchases",
                   "NumWebVisitsMonth", "Kidhome", "Teenhome"]

    missing_cols = [col for col in key_columns if col not in df.columns]
    if missing_cols:
        st.error(f"The uploaded CSV is missing the following required columns: {', '.join(missing_cols)}")
        st.stop()

    numeric_cols = key_columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=key_columns)

    if df.empty:
        st.error("No valid data found after removing rows with missing values.")
        st.stop()

    # Feature Engineering
    df["Age"] = 2025 - df["Year_Birth"]

    df["Total_Spending"] = (
        df["MntWines"]
        + df["MntFruits"]
        + df["MntMeatProducts"]
        + df["MntFishProducts"]
        + df["MntSweetProducts"]
        + df["MntGoldProds"]
    )

    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

    features = [
        "Income",
        "Recency",
        "Total_Spending",
        "NumWebPurchases",
        "NumStorePurchases",
        "NumCatalogPurchases",
        "NumWebVisitsMonth",
        "Age",
        "Total_Children"
    ]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])

    # Elbow Method
    wcss = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(scaled_features)
        wcss.append(km.inertia_)

    # Final Model
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)
    df["Cluster"] = clusters

    silhouette = silhouette_score(scaled_features, clusters)

    # Sidebar Filters
    st.sidebar.header("Filters")

    age_range = st.sidebar.slider(
        "Age Range",
        int(df["Age"].min()),
        int(df["Age"].max()),
        (int(df["Age"].min()), int(df["Age"].max()))
    )

    income_range = st.sidebar.slider(
        "Income Range",
        int(df["Income"].min()),
        int(df["Income"].max()),
        (int(df["Income"].min()), int(df["Income"].max()))
    )

    selected_clusters = st.sidebar.multiselect(
        "Select Clusters",
        options=sorted(df["Cluster"].unique()),
        default=sorted(df["Cluster"].unique())
    )

    filtered_df = df[
        (df["Age"] >= age_range[0]) &
        (df["Age"] <= age_range[1]) &
        (df["Income"] >= income_range[0]) &
        (df["Income"] <= income_range[1]) &
        (df["Cluster"].isin(selected_clusters))
    ]

    if filtered_df.empty:
        st.warning("No data matches the selected filters. Please adjust your filters.")
    else:
        # ✅ FIXED: 5 columns defined to match 5 metric calls
        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Customers", len(filtered_df))
        c2.metric("Clusters", 4)
        c3.metric("Selected Clusters", len(selected_clusters))
        c4.metric("Silhouette Score", round(silhouette, 3))
        c5.metric("Avg Spending", f"${int(filtered_df['Total_Spending'].mean())}")

        st.divider()

        # Elbow Chart
        st.subheader("Optimal Cluster Analysis")

        elbow_fig = go.Figure()
        elbow_fig.add_trace(
            go.Scatter(x=list(range(2, 9)), y=wcss, mode="lines+markers")
        )
        elbow_fig.update_layout(
            template="plotly_dark",
            title="Elbow Method (WCSS)",
            xaxis_title="Number of Clusters",
            yaxis_title="WCSS"
        )
        st.plotly_chart(elbow_fig, use_container_width=True)

        # PCA Visualization
        st.subheader("Customer Segmentation Visualization")

        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaled_features)

        viz_df = pd.DataFrame({
            "PCA1": reduced_data[:, 0],
            "PCA2": reduced_data[:, 1],
            "Cluster": clusters.astype(str)
        })

        scatter_fig = px.scatter(
            viz_df,
            x="PCA1",
            y="PCA2",
            color="Cluster",
            template="plotly_dark",
            title="PCA-Based Customer Cluster Visualization"
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

        # Cluster Distribution
        st.subheader("Cluster Distribution")

        pie_fig = px.pie(
            filtered_df,
            names="Cluster",
            template="plotly_dark",
            title="Customer Distribution Across Clusters"
        )
        st.plotly_chart(pie_fig, use_container_width=True)

        # Cluster Summary
        st.subheader("Cluster Intelligence Summary")

        summary = filtered_df.groupby("Cluster")[
            [
                "Income",
                "Total_Spending",
                "NumWebPurchases",
                "NumStorePurchases",
                "NumCatalogPurchases",
                "NumWebVisitsMonth",
                "Age"
            ]
        ].mean().round(2)

        st.dataframe(summary, use_container_width=True)

        # Dynamic Business Insights
        st.subheader("AI Business Recommendations")

        highest_spending_cluster = summary["Total_Spending"].idxmax()
        highest_web_cluster = summary["NumWebVisitsMonth"].idxmax()
        highest_income_cluster = summary["Income"].idxmax()
        churn_candidates = summary["NumWebVisitsMonth"].idxmin()

        st.success(
            f"Cluster {highest_spending_cluster}: High-value premium customers. "
            f"Target with loyalty rewards, VIP discounts, and premium offers."
        )
        st.info(
            f"Cluster {highest_web_cluster}: High browsing behavior with weaker conversions. "
            f"Retarget with personalized coupons and remarketing campaigns."
        )
        st.warning(
            f"Cluster {highest_income_cluster}: Strong purchasing potential. "
            f"Upsell premium products and cross-sell recommendations."
        )
        st.error(
            f"Cluster {churn_candidates}: Possible churn-risk users. "
            f"Launch win-back campaigns and engagement emails."
        )

        # Dataset Preview
        st.subheader("Filtered Dataset Preview")
        st.dataframe(filtered_df.head(20), use_container_width=True)

        # CSV Export
        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Clustered Customer Report",
            data=csv,
            file_name="smartcart_clustered_customers.csv",
            mime="text/csv"
        )

else:
    st.info("Upload your SmartCart customer dataset CSV to begin.")
