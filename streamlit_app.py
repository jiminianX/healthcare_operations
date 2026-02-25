## Step 00 - Import of the packages

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import sklearn

from ydata_profiling import ProfileReport
from streamlit.components.v1 import html

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
# from streamlit_pandas_profiling import st_profile_report

st.set_page_config(
    page_title="Healthcare Clinic Revenue Dashboard 🏥",
    layout="centered",
    page_icon="🏥",
)


## Step 01 - Setup
st.sidebar.title("Healthcare Clinics 🏥")
page = st.sidebar.selectbox("Select Page",["Business Case 📘","Visualization 📊", "Automated Report 📄", "Prediction"])


#st.video("video.mp4")

st.image("hospital.jpg")

st.write("   ")
st.write("   ")
st.write("   ")
patients = pd.read_csv("patients.csv")
appointments = pd.read_csv("appointments.csv")
clinics = pd.read_csv("clinics.csv")

df = appointments.merge(patients, on="patient_id", how="left")
df = df.merge(clinics, on="clinic_id", how="left")

df.head()  

## Step 02 - Load dataset
if page == "Business Case 📘":

    st.subheader("Healthcare Clinic Revenue Dashboard")

    st.markdown("""
    ## 🎯 Business Problem

    Healthcare clinics lose revenue due to:
    - Patient no-shows
    - Underutilized appointment slots
    - Inefficient scheduling
    - Poor resource allocation

    These inefficiencies lead to hidden financial losses.
    """)

    st.markdown("""
    ## Our Solution

    1. Identify revenue loss per clinic  
    2. Predict no-show rates using Linear Regression  
    3. Recommend optimized scheduling strategies  
    """)

    st.markdown("##### Data Preview")
    rows = st.slider("Select a number of rows to display",5,20,5)
    st.dataframe(patients.head(rows))
    st.dataframe(appointments.head(rows))
    st.dataframe(clinics.head(rows))

    st.markdown("##### Missing values")
    missing_patients = patients.isnull().sum()
    missing_appointments = appointments.isnull().sum()
    missing_clinics = clinics.isnull().sum()

    st.write(missing_patients)
    st.write(missing_appointments)
    st.write(missing_clinics)

    st.markdown("##### Data Shape")
    st.write("Patients:", patients.shape)
    st.write("Appointments:", appointments.shape)
    st.write("Clinics:", clinics.shape)

    if missing_patients.sum() == 0 and missing_appointments.sum() == 0 and missing_clinics.sum() == 0:
        st.success("✅ No missing values found")
    else:
        st.warning("⚠️ you have missing values")

     

    st.markdown("##### 📈 Summary Statistics")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())

elif page == "Visualization 📊":

    ## Step 03 - Data Viz
    st.subheader("02 Data Vizualization")

    # col_x = st.selectbox("Select X-axis variable", df.columns, index=0)
    # col_y = st.selectbox("Select Y-axis variable", df.columns, index=1)

    tab1, tab2, tab3, tab4 = st.tabs(["Revenue Leakage 📊", "Patient Lifetime Value 💰", "Correlation Matrix 🔥" ,"No-Show Rate by Clinic"])

    with tab1:
        st.subheader("Revenue Leakage Bar Chart")
        df['revenue_loss'] = df['revenue_expected'] - df['revenue_realized']
        revenue_by_clinic = df.groupby("clinic_location").agg(
            expected_revenue=('revenue_expected', 'sum'),
            realized_revenue=('revenue_realized', 'sum'),
            revenue_loss=('revenue_loss', 'sum')
        ).reset_index()
        revenue_by_clinic['revenue_loss_percent'] = (
            revenue_by_clinic['revenue_loss'] / revenue_by_clinic['expected_revenue'] * 100
        )
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.barplot(data=revenue_by_clinic, x='clinic_location', y='revenue_loss_percent', ax=ax1)
        ax1.set_title("Revenue Loss % by Clinic", fontsize=16)
        ax1.set_ylabel("% Revenue Lost")
        ax1.set_xlabel("Clinic")
        st.pyplot(fig1)

    with tab2:
        st.subheader("Patient Lifetime Value")
        ltv = patients.copy()
        ltv['expected_future_visits'] = 10 - ltv['total_lifetime_visits']
        ltv['estimated_LTV'] = (
            ltv['total_lifetime_revenue']
            + (ltv['expected_future_visits'] * ltv['total_lifetime_revenue']
            / ltv['total_lifetime_visits'].replace(0, 1))
        )
        st.dataframe(ltv.head())


    with tab3:
        st.subheader("Correlation Matrix")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            df[['age', 'chronic_condition_flag', 'total_lifetime_visits',
                'revenue_expected', 'revenue_realized', 'no_show_flag']].corr(),
            annot=True,
            cmap='Blues',
            fmt=".2f",
            ax=ax3
        )
        ax3.set_title("Correlation Heatmap")
        st.pyplot(fig3)

    with tab4:
        st.subheader("No-Show Rate by Clinic")
        no_show_by_clinic = df.groupby("clinic_location")["no_show_flag"].mean().reset_index()
        no_show_by_clinic.rename(columns={"no_show_flag": "no_show_rate"}, inplace=True)
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        sns.barplot(data=no_show_by_clinic, x='clinic_location', y='no_show_rate', ax=ax4)
        ax4.set_title("No-Show Rate by Clinic", fontsize=16)
        ax4.set_ylabel("No-show Rate")
        ax4.set_xlabel("Clinic")
        st.pyplot(fig4)

elif page == "Automated Report 📑":
    st.subheader("03 Automated Report")
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            profile = ProfileReport(df,title="Clinic Revenue Report",explorative=True,minimal=True)
            html(profile.to_html(), height=1000)

        export = profile.to_html()
        st.download_button(label="📥 Download full Report",data=export,file_name="clinic_revenue_report.html",mime='text/html')


elif page == "Prediction":
    st.subheader("04 Prediction with Linear Regression")
    df2 = df
    ## Data Preprocessing

    ### removing missing values 
    df2 = df2.dropna()

    ### Label Encoder to change text categories into number categories
    
    le = LabelEncoder()

    df2["ocean_proximity"] = le.fit_transform(df2["ocean_proximity"])

    list_var = list(df2.columns)

    features_selection = st.sidebar.multiselect("Select Features (X)",list_var,default=list_var)
    target_selection  = st.sidebar.selectbox("Select target variable (Y))",list_var)
    selected_metrics = st.sidebar.multiselect("Metrics to display", ["Mean Squared Error (MSE)","Mean Absolute Error (MAE)","R² Score"],default=["Mean Absolute Error (MAE)"])

    ### i) X and y
    X = df2[features_selection]
    y = df2[target_selection]

    st.dataframe(X.head())
    st.dataframe(y.head())

    ### ii) train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


    ## Model 

    ### i) Definition model
    model = LinearRegression()

    ### ii) Training model
    model.fit(X_train,y_train)

    ### iii) Prediction
    predictions = model.predict(X_test)

    ### iv) Evaluation  
    if "Mean Squared Error (MSE)" in selected_metrics:
        mse = metrics.mean_squared_error(y_test, predictions)
        st.write(f"- **MSE** {mse:,.2f}")
    if "Mean Absolute Error (MAE)" in selected_metrics:
        mae = metrics.mean_absolute_error(y_test, predictions)
        st.write(f"- **MAE** {mae:,.2f}")
    if "R² Score" in selected_metrics:
        r2 = metrics.r2_score(y_test, predictions)
        st.write(f"- **R2** {r2:,.3f}")

    st.success(f"My model performance is of ${np.round(mae,2)}")

    fig, ax = plt.subplots()
    ax.scatter(y_test,predictions,alpha=0.5)
    ax.plot([y_test.min(),y_test.max()],
           [y_test.min(),y_test.max() ],"--r",linewidth=2)
    ax.set_xlabel("Actual Value")
    ax.set_ylabel("Predicted Value")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)