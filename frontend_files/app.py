
import os
import requests
import pandas as pd
import streamlit as st

# Address of the Flask backend. Inside the Docker network the backend container is reachable by its name.
BACKEND_URL = os.getenv("BACKEND_URL", "http://superkart-backend:7860").rstrip("/")

st.set_page_config(page_title="SuperKart Sales Forecaster", page_icon="🛒", layout="centered")
st.title("🛒 SuperKart Sales Forecaster")
st.write(
    "Forecast the total revenue a product will generate in a given SuperKart store. "
    "Fill in the product and store details, or upload a CSV file to forecast many products at once."
)

tab_single, tab_batch = st.tabs(["Single prediction", "Batch prediction"])

# ------------------------------------------------------------------ single prediction
with tab_single:
    st.subheader("Product details")
    col1, col2 = st.columns(2)
    with col1:
        product_weight = st.number_input("Product weight", min_value=1.0, max_value=30.0, value=12.65, step=0.1)
        product_mrp = st.number_input("Product MRP (maximum retail price)", min_value=1.0, max_value=400.0, value=147.0, step=1.0)
        allocated_area = st.number_input("Allocated display area (ratio)", min_value=0.001, max_value=0.5, value=0.056, step=0.001, format="%.3f")
    with col2:
        sugar = st.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
        product_family = st.selectbox("Product family (Product_Id prefix)", ["FD", "DR", "NC"],
                                      help="FD = food, DR = drinks, NC = non-consumable")
        type_category = st.selectbox("Product type category", ["Perishables", "Non Perishables"])

    st.subheader("Store details")
    col3, col4 = st.columns(2)
    with col3:
        store_size = st.selectbox("Store size", ["Small", "Medium", "High"], index=1)
        city_type = st.selectbox("City tier", ["Tier 1", "Tier 2", "Tier 3"], index=1)
    with col4:
        store_type = st.selectbox("Store type", ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"], index=2)
        store_age = st.number_input("Store age (years)", min_value=0, max_value=60, value=16, step=1)

    if st.button("Forecast sales", type="primary"):
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": sugar,
            "Product_Allocated_Area": allocated_area,
            "Product_MRP": product_mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": city_type,
            "Store_Type": store_type,
            "Product_Id_char": product_family,
            "Store_Age_Years": int(store_age),
            "Product_Type_Category": type_category,
        }
        try:
            response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload, timeout=30)
            if response.status_code == 200:
                st.success(f"Forecasted sales: **{response.json()['Predicted_Sales']:,.2f}**")
            else:
                st.error(f"The API returned an error ({response.status_code}): {response.text}")
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not reach the backend at {BACKEND_URL}: {exc}")

# ------------------------------------------------------------------ batch prediction
with tab_batch:
    st.subheader("Upload a CSV file")
    st.write(
        "The file needs these columns: `Product_Weight`, `Product_Sugar_Content`, `Product_Allocated_Area`, "
        "`Product_MRP`, `Store_Size`, `Store_Location_City_Type`, `Store_Type`, `Product_Id_char`, "
        "`Store_Age_Years`, `Product_Type_Category`."
    )
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        st.write("Preview of the uploaded data:")
        st.dataframe(batch_df.head())
        if st.button("Forecast sales for the whole file"):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/v1/predictbatch",
                    files={"file": ("batch.csv", batch_df.to_csv(index=False).encode("utf-8"))},
                    timeout=60,
                )
                if response.status_code == 200:
                    predictions = response.json()
                    values = [predictions[str(i)] for i in batch_df.index]
                    result = batch_df.copy()
                    # Rejected rows come back as text explaining the problem; keep it in its own column
                    result["Predicted_Sales"] = [v if isinstance(v, (int, float)) else None for v in values]
                    result["Issue"] = [v if isinstance(v, str) else "" for v in values]
                    rejected = int((result["Issue"] != "").sum())
                    if rejected:
                        st.warning(f"{rejected} row(s) were rejected and have no forecast. See the Issue column.")
                    st.dataframe(result)
                    st.download_button("Download predictions", result.to_csv(index=False).encode("utf-8"),
                                       file_name="superkart_predictions.csv", mime="text/csv")
                else:
                    st.error(f"The API returned an error ({response.status_code}): {response.text}")
            except requests.exceptions.RequestException as exc:
                st.error(f"Could not reach the backend at {BACKEND_URL}: {exc}")
