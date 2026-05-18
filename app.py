import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="K-Means Customer Segmentation",
    layout="centered"
)

st.title("K-Means Customer Segmentation")
st.write("This app groups customers based on annual income and spending score.")

df = pd.read_csv("Mall_Customers.csv")

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Enter Customer Details")

income = st.number_input(
    "Annual Income (k$)",
    min_value=0,
    max_value=200,
    value=60
)

score = st.number_input(
    "Spending Score (1-100)",
    min_value=1,
    max_value=100,
    value=50
)

if st.button("Predict Cluster"):
    input_data = pd.DataFrame(
        [[income, score]],
        columns=['Annual Income (k$)', 'Spending Score (1-100)']
    )

    input_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_scaled)[0]

    st.success(f"The customer belongs to Cluster {cluster}")

    if cluster == 0:
        st.write("This customer belongs to segment 0.")
    elif cluster == 1:
        st.write("This customer belongs to segment 1.")
    elif cluster == 2:
        st.write("This customer belongs to segment 2.")
    elif cluster == 3:
        st.write("This customer belongs to segment 3.")
    else:
        st.write("This customer belongs to segment 4.")

st.subheader("Customer Segments")

fig, ax = plt.subplots(figsize=(8, 5))

sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    data=df,
    palette='Set1',
    s=80,
    ax=ax
)

plt.title("Customer Segments")
st.pyplot(fig)