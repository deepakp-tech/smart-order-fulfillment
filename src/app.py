import streamlit as st
import sqlite3
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from geopy.distance import geodesic

st.set_page_config(page_title="Smart Order Fulfillment", page_icon="📦", layout="wide")

# 1. LOAD ML MODEL & ASSETS
try:
    model_pkg = joblib.load("../models/smart_fulfillment_stack.pkl")
    model = model_pkg['model']
    le = model_pkg['label_encoder']
    u_map = model_pkg['urgency_map']
    # Load warehouse data for coordinates
    warehouses = pd.read_csv("../data/warehouses.csv")
except Exception as e:
    st.error("⚠️ Error loading model or warehouse data. Make sure 'smart_fulfillment_stack.pkl' and 'warehouses.csv' exist.")

# 2. DATABASE SETUP
conn = sqlite3.connect("orders.db", check_same_thread=False)
cursor = conn.cursor()

# Updated schema to include location and assignment data
cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer TEXT,
        product TEXT,
        quantity INTEGER,
        priority TEXT,
        latitude REAL,
        longitude REAL,
        assigned_warehouse TEXT,
        distance_km REAL
    )
""")
conn.commit()

# 3. UI HEADER
st.markdown("<h1 style='text-align: center;'>📦 Smart Order Fulfillment System</h1>", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["➕ Add Order", "📋 Order Registry", "📈 Business Analytics"])

# --- TAB 1: Add Order & Real-Time Fulfillment ---
with tab1:
    st.subheader("➕ Create New Fulfillment Request")
    
    with st.form("order_form"):
        col1, col2 = st.columns(2)
        with col1:
            customer = st.text_input("👤 Customer Name")
            quantity = st.number_input("🔢 Quantity", min_value=1, value=1)
            lat = st.number_input("📍 Customer Latitude (e.g., 28.6139)", format="%.6f")
        with col2:
            product = st.text_input("📦 Product Name")
            priority = st.selectbox("⚡ Priority Level", ["High", "Medium", "Low"])
            lon = st.number_input("📍 Customer Longitude (e.g., 77.2090)", format="%.6f")

        submitted = st.form_submit_button("🚀 Assign Warehouse & Submit")

        if submitted:
            if customer.strip() == "" or product.strip() == "":
                st.warning("⚠️ Please fill in all name fields.")
            else:
                # RUN PREDICTION
                input_features = pd.DataFrame([[
                    lat, lon, quantity, u_map[priority]
                ]], columns=["latitude", "longitude", "quantity", "urgency_rank"])
                
                pred_idx = model.predict(input_features)
                assigned_wh = le.inverse_transform(pred_idx)[0]

                # CALCULATE REAL DISTANCE
                wh_coords = warehouses[warehouses["name"] == assigned_wh][["latitude", "longitude"]].values[0]
                dist = geodesic((lat, lon), wh_coords).km

                # SAVE TO DB
                cursor.execute("""
                    INSERT INTO orders (customer, product, quantity, priority, latitude, longitude, assigned_warehouse, distance_km) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (customer, product, quantity, priority, lat, lon, assigned_wh, dist))
                conn.commit()
                
                st.success(f"✅ Order for {customer} assigned to **{assigned_wh}**")
                st.info(f"📏 Fulfillment Distance: **{dist:.2f} km**")

# --- TAB 2: View Orders ---
with tab2:
    st.subheader("📋 Complete Fulfillment History")
    df = pd.read_sql_query("SELECT * FROM orders", conn)

    if df.empty:
        st.info("No orders processed yet.")
    else:
        st.dataframe(df.style.highlight_max(axis=0, subset=['distance_km']), use_container_width=True)

# --- TAB 3: Analytics ---
with tab3:
    st.subheader("📈 Performance Insights")
    df = pd.read_sql_query("SELECT * FROM orders", conn)

    if df.empty:
        st.warning("No data available for analysis.")
    else:
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### ⚡ Priority Distribution")
            priority_counts = df["priority"].value_counts()
            fig, ax = plt.subplots()
            ax.pie(priority_counts, labels=priority_counts.index, autopct='%1.1f%%', startangle=90)
            st.pyplot(fig)

        with c2:
            st.markdown("#### 🚛 Avg Distance by Priority")
            avg_dist = df.groupby("priority")["distance_km"].mean()
            st.bar_chart(avg_dist)

        st.markdown("#### 🏢 Warehouse Workload")
        wh_counts = df["assigned_warehouse"].value_counts()
        st.bar_chart(wh_counts)
