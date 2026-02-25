## Step 00 - Import of the packages

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import sklearn

from ydata_profiling import ProfileReport
from streamlit.components.v1 import html
# from streamlit_pandas_profiling import st_profile_report

st.set_page_config(
    page_title="Healthcare Clinic Revenue Dashboard 🏥",
    layout="centered",
    page_icon="🏥",
)


## Step 01 - Setup
st.sidebar.title("Healthcare Clinics 🏥")
page = st.sidebar.selectbox("Select Page",["Introduction 📘","Visualization 📊", "Automated Report 📑","Prediction"])


#st.video("video.mp4")

st.image("https://media.istockphoto.com/id/1524913019/fr/vectoriel/illustration-vectorielle-de-b%C3%A2timent-hospitalier-dans-un-design-plat.jpg?s=612x612&w=0&k=20&c=-hJTmokrmJ2nnmJoANeSkAAwmyrScKZp75jD6D8OIII=")

st.write("   ")
st.write("   ")
st.write("   ")
patients = pd.read_csv("patients.csv")
appointments = pd.read_csv("appointments.csv")
clinics = pd.read_csv("clinics.csv")

appt_patient = appointments.merge(patients, on="patient_id", how="left")
df = appt_patient.merge(clinics, on="clinic_id", how="left")

df.head()  

## Step 02 - Load dataset
if page == "Introduction 📘":

    st.subheader("01 Introduction 📘")

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

    if missing_patients.sum() == 0 or missing_appointments == 0 or missing_clinics == 0:
        st.success("✅ No missing values found")
    else:
        st.warning("⚠️ you have missing values")

     

    st.markdown("##### 📈 Summary Statistics")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())

elif page == "Visualization 📊":

    ## Step 03 - Data Viz
    st.subheader("02 Data Viz")

    col_x = st.selectbox("Select X-axis variable", df.columns, index=0)
    col_y = st.selectbox("Select Y-axis variable", df.columns, index=1)

    tab1, tab2, tab3 = st.tabs(["Bar Chart 📊","Line Chart 📈","Correlation Heatmap 🔥"])

    with tab1:
        st.subheader("Bar Chart")
        st.bar_chart(df[[col_x,col_y]].sort_values(by=col_x),use_container_width=True)

    with tab2:
        st.subheader("Line Chart")
        st.line_chart(df[[col_x,col_y]].sort_values(by=col_x),use_container_width=True)


    with tab3:
        st.subheader("Correlation Matrix")
        df_numeric = df.select_dtypes(include=np.number)

        fig_corr, ax_corr = plt.subplots(figsize=(18,14))
        # create the plot, in this case with seaborn 
        sns.heatmap(df_numeric.corr(),annot=True,fmt=".2f",cmap='coolwarm')
        ## render the plot in streamlit 
        st.pyplot(fig_corr)

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
    from sklearn.preprocessing import LabelEncoder
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
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


    ## Model 

    ### i) Definition model
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()

    ### ii) Training model
    model.fit(X_train,y_train)

    ### iii) Prediction
    predictions = model.predict(X_test)

    ### iv) Evaluation 
    from sklearn import metrics 
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
    ax.set_xlabel("Predicted Value")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)