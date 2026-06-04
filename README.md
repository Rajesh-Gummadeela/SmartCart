# 🛒 SmartCart AI — Customer Intelligence Dashboard

> AI-powered customer segmentation dashboard that helps businesses identify high-value customers, churn risks, and conversion opportunities — built with Streamlit, scikit-learn, and Plotly.

---

## 📸 Overview

SmartCart AI transforms raw customer data into actionable business intelligence using unsupervised machine learning. Upload a customer dataset, and the dashboard automatically segments your customers into behavioral clusters, visualizes the results, and generates targeted business recommendations — all in an interactive web UI.

---

## ✨ Features

- **Automated K-Means Clustering** — segments customers into 4 behavioral groups based on spending, income, recency, and purchase channel
- **Elbow Method Analysis** — visualizes WCSS across k=2 to k=8 to validate the optimal cluster count
- **PCA Visualization** — reduces high-dimensional features to 2D for intuitive cluster scatter plots
- **Silhouette Score** — quantifies cluster quality at a glance
- **Interactive Filters** — filter by age range, income range, and specific clusters in real time
- **AI Business Recommendations** — auto-generated insights targeting high-value customers, browsing drop-offs, upsell candidates, and churn risks
- **CSV Export** — download the clustered dataset for use in downstream tools

---

## 🧠 How It Works

1. Customer data is loaded and cleaned
2. Key features are engineered (total spending, age, total children)
3. Features are scaled with `StandardScaler`
4. K-Means clustering (`k=4`) assigns each customer to a segment
5. PCA reduces the feature space to 2 components for visualization
6. The dashboard surfaces cluster summaries and business insights

**Features used for clustering:**

| Feature | Description |
|---|---|
| `Income` | Annual household income |
| `Recency` | Days since last purchase |
| `Total_Spending` | Sum of spend across all product categories |
| `NumWebPurchases` | Online purchase count |
| `NumStorePurchases` | In-store purchase count |
| `NumCatalogPurchases` | Catalog purchase count |
| `NumWebVisitsMonth` | Monthly website visits |
| `Age` | Derived from `Year_Birth` |
| `Total_Children` | Kids + teenagers at home |

---

## 📁 Dataset Requirements

Upload a `.csv` file with the following columns:

```
Income, Year_Birth, MntWines, MntFruits, MntMeatProducts,
MntFishProducts, MntSweetProducts, MntGoldProds, Recency,
NumWebPurchases, NumStorePurchases, NumCatalogPurchases,
NumWebVisitsMonth, Kidhome, Teenhome
```

> Compatible with the [UCI Marketing Campaign Dataset](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis).

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/smartcart-ai.git
cd smartcart-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run smartcart_dashboard.py
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
plotly
```

Or install directly:

```bash
pip install streamlit pandas numpy scikit-learn plotly
```

---

## 📊 Dashboard Sections

| Section | Description |
|---|---|
| **KPI Metrics** | Customers, clusters, silhouette score, average spending |
| **Elbow Chart** | WCSS curve to validate k=4 selection |
| **PCA Scatter Plot** | 2D cluster visualization via principal component analysis |
| **Cluster Distribution** | Pie chart of customer share per cluster |
| **Cluster Summary Table** | Mean stats per cluster across all features |
| **AI Recommendations** | Auto-generated strategy cards per cluster |
| **Dataset Preview** | Filtered raw data with cluster labels |
| **CSV Export** | Download the segmented dataset |

---

## 🗂 Project Structure

```
smartcart-ai/
│
├── smartcart_dashboard.py   # Main Streamlit app
├── requirements.txt         # Python dependencies
└── README.md
```

---

<p align="center">Built with using Streamlit · scikit-learn · Plotly</p>
