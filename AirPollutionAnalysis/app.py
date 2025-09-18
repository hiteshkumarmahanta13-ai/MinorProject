import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk
import os

DATA_PATH1 = os.path.join(os.path.dirname(__file__), "data", "bbsr_air24.csv")
DATA_PATH2 = os.path.join(os.path.dirname(__file__), "data", "delhi_mly.csv")


# ==============================
# Page Config
# ==============================
# Use a wide layout for better side-by-side visualization
st.set_page_config(layout="wide")




# ==============================
# Load Data
# ==============================
@st.cache_data
def load_data():
    bbsr_df = pd.read_csv(DATA_PATH1)
    delhi_df = pd.read_csv(DATA_PATH2)

    # Detect datetime if available
    for df in [bbsr_df, delhi_df]:
        datetime_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        if datetime_cols:
            df[datetime_cols[0]] = pd.to_datetime(df[datetime_cols[0]], errors="coerce")
            df.sort_values(by=datetime_cols[0], inplace=True)
    return bbsr_df, delhi_df

bbsr_df, delhi_df = load_data()

st.title("Air Pollution Analysis")
st.markdown("Interactive dashboard to explore air pollution patterns in *Bhubaneswar* and *Delhi*.")

# ==============================
# Sidebar Filters
# ==============================
st.sidebar.header("🔍 Filters")
city = st.sidebar.selectbox("Select City", ["Bhubaneswar", "Delhi"])

if city == "Delhi":
    year = st.sidebar.selectbox("Select Year", sorted(delhi_df["year"].dropna().unique()))
    pollutants = [col for col in delhi_df.columns if col not in ["year", "month", "AQI", "city"]]
    selected_pollutants = st.sidebar.multiselect("Select Pollutants", pollutants, default=pollutants[:3])
    filtered_df = delhi_df[delhi_df["year"] == year].copy() # Use .copy() to avoid SettingWithCopyWarning
    date_col = [c for c in delhi_df.columns if 'date' in c.lower() or 'time' in c.lower()]
elif city == "Bhubaneswar":
    pollutants = [col for col in bbsr_df.columns if col not in ["month", "hour", "dayofweek"]]
    selected_pollutants = st.sidebar.multiselect("Select Pollutants", pollutants, default=pollutants[:3])
    filtered_df = bbsr_df.copy() # Use .copy() to avoid SettingWithCopyWarning
    date_col = [c for c in bbsr_df.columns if 'date' in c.lower() or 'time' in c.lower()]

date_col = date_col[0] if date_col else None


# ==============================
# KPI Cards & Composition Charts
# ==============================
st.subheader("Key Performance Indicator")
col1, col2, col3 = st.columns([1.5, 2, 2]) # Adjust column ratios for better fit

with col1:
    st.markdown("###### *Key Metrics*")
    if selected_pollutants:
        numeric_df = filtered_df[selected_pollutants].apply(pd.to_numeric, errors="coerce")
        st.metric("Max Value", f"{numeric_df.max().max():.2f}")
        st.metric("Average Value", f"{numeric_df.mean().mean():.2f}")
        st.metric("Alerts (>100)", f"{(numeric_df > 100).sum().sum()}")
    else:
        st.info("Select pollutants.")

with col2:
    if selected_pollutants:
        averages = filtered_df[selected_pollutants].apply(pd.to_numeric, errors="coerce").mean()
        pie_fig = px.pie(values=averages.values, names=averages.index, title="Pollutant Composition (Pie)")
        pie_fig.update_traces(textposition='inside', textinfo='percent+label')
        pie_fig.update_layout(margin=dict(l=0, r=0, t=30, b=0)) # Reduce margins
        st.plotly_chart(pie_fig, use_container_width=True)
    else:
        st.warning("⚠ Select pollutants.")

with col3:
    if selected_pollutants:
        averages = filtered_df[selected_pollutants].apply(pd.to_numeric, errors="coerce").mean()
        doughnut_fig = px.pie(values=averages.values, names=averages.index, hole=0.6, title="Pollutant Composition (Doughnut)")
        doughnut_fig.update_traces(textposition='inside', textinfo='percent+label')
        doughnut_fig.update_layout(margin=dict(l=0, r=0, t=30, b=0)) # Reduce margins
        st.plotly_chart(doughnut_fig, use_container_width=True)
    else:
        st.warning("⚠ Select pollutants.")


# ==============================
# Line Chart & Heatmap
# ==============================
col1, col2 = st.columns(2)

with col1:
    # Line Chart (Month-wise or Time-series)
    if selected_pollutants:
        if "month" in filtered_df.columns:
            st.subheader("Monthly Trend")
            month_map = {
                1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun", 7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec",
                "January":"Jan","February":"Feb","March":"Mar","April":"Apr", "May":"May","June":"Jun","July":"Jul","August":"Aug",
                "September":"Sep","October":"Oct","November":"Nov","December":"Dec"
            }
            # Normalize month column without raising SettingWithCopyWarning
            filtered_df["month_norm"] = filtered_df["month"].map(month_map).fillna(filtered_df["month"])
            month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            monthwise_line = (
                filtered_df[selected_pollutants].apply(pd.to_numeric, errors="coerce")
                .groupby(filtered_df["month_norm"]).mean()
                .reindex(month_order).dropna(how="all")
            )
            if not monthwise_line.empty:
                fig_line = px.line(monthwise_line, x=monthwise_line.index, y=selected_pollutants, markers=True)
                fig_line.update_layout(title="Pollutant Concentrations vs Month")
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info("No data for this trend.")
        elif date_col:
            st.subheader("Time Series Trend")
            fig_line = px.line(filtered_df, x=date_col, y=selected_pollutants, title="Pollutant Concentrations Over Time")
            st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.warning("⚠ Select pollutants to see trends.")

with col2:
    st.subheader("Correlation Heatmap")
    all_pollutants = [col for col in (bbsr_df if city=="Bhubaneswar" else delhi_df).columns if col not in ["month_norm","year","month","dayofweek", "hour", "AQI", "city", date_col]]
    numeric_all = (bbsr_df if city=="Bhubaneswar" else delhi_df)[all_pollutants].apply(pd.to_numeric, errors="coerce")
    corr = numeric_all.corr()
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

# ==============================
# Boxplot & Scatterplot
# ==============================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Pollutant Distribution")
    if selected_pollutants:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.boxplot(data=filtered_df[selected_pollutants].apply(pd.to_numeric, errors="coerce"), ax=ax)
        st.pyplot(fig)
    else:
        st.info("Select pollutants to see boxplots.")

with col2:
    st.subheader("Pollutant Relationship")
    if len(selected_pollutants) == 2:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.regplot(
            x=pd.to_numeric(filtered_df[selected_pollutants[0]], errors="coerce"),
            y=pd.to_numeric(filtered_df[selected_pollutants[1]], errors="coerce"),
            scatter_kws={"alpha": 0.6}, ax=ax
        )
        st.pyplot(fig)
    else:
        st.info("Select exactly two pollutants to see a scatterplot.")

# ==============================
# Map & Alerts
# ==============================
col1, col2 = st.columns([3, 2]) # Give more width to the map

with col1:
    st.subheader(f"🗺 Pollution Map of {city}")
    if {'lat','lon'}.issubset(filtered_df.columns) or {'latitude','longitude'}.issubset(filtered_df.columns):
        lat_col = 'lat' if 'lat' in filtered_df.columns else 'latitude'
        lon_col = 'lon' if 'lon' in filtered_df.columns else 'longitude'
    else:
        np.random.seed(42)
        if city == "Bhubaneswar":
            filtered_df['lat'] = 20.2961 + np.random.uniform(-0.02, 0.02, len(filtered_df))
            filtered_df['lon'] = 85.8245 + np.random.uniform(-0.02, 0.02, len(filtered_df))
        else: # Delhi
            filtered_df['lat'] = 28.7041 + np.random.uniform(-0.05, 0.05, len(filtered_df))
            filtered_df['lon'] = 77.1025 + np.random.uniform(-0.05, 0.05, len(filtered_df))
        lat_col, lon_col = 'lat', 'lon'

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(latitude=filtered_df[lat_col].mean(), longitude=filtered_df[lon_col].mean(), zoom=10),
        layers=[
            pdk.Layer('ScatterplotLayer', data=filtered_df, get_position='[lon, lat]',
                      get_radius=120, get_fill_color='[255, 0, 0, 140]', pickable=True)
        ]
    ))

with col2:
    st.subheader("Alerts (>100)")
    if selected_pollutants:
        numeric_df = filtered_df[selected_pollutants].apply(pd.to_numeric, errors="coerce")
        alerts = numeric_df[(numeric_df > 100).any(axis=1)]
        if not alerts.empty:
            st.warning(f"{len(alerts)} records exceed safe limits.")
            display_cols = selected_pollutants
            if date_col:
                display_cols = [date_col] + selected_pollutants
            st.dataframe(filtered_df.loc[alerts.index, display_cols])
        else:
            st.success("No alerts found.")
    else:

        st.info("Select pollutants to check for alerts.")

