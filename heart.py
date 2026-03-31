import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

st.markdown("""
<style>

/* Main background (soft gradient medical theme) */
.stApp {
    background: linear-gradient(to right,#e66465, #9198e5);
}

/* Title styling */
h1 {
    color: #0242a8;
    text-align: center;
    font-weight: 700;
}

/* Tabs styling */
button[data-baseweb="tab"] {
    background-color: #dce6ff;
    border-radius: 10px;
    margin: 5px;
    padding: 10px;
    font-weight: bold;
    color: #1f3b73;
}

/* Active tab highlight */
button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #a9c1ff;
}

/* Input boxes */
div[data-baseweb="input"] {
    background-color: #ffffff;
    border-radius: 12px;
    border: 1px solid #c7d4ff;
}

/* Dropdown select boxes */
div[data-baseweb="select"] {
    background-color: #ffffff;
    border-radius: 12px;
    border: 1px solid #c7d4ff;
}

/* Buttons */
.stButton > button {
    background-color :#0ac1c7;
    color: white;
    border-radius: 12px;
    height: 48px;
    width: 220px;
    font-size: 16px;
    font-weight: bold;
}

/* Info box styling */
.stInfo {
    background-color: #e8f0ff;
    border-radius: 10px;
}

/* Success styling */
.stSuccess {
    background-color: #e6f7f0;
    border-radius: 10px;
}

/* Warning styling */
.stWarning {
    background-color: #fff2e6;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


def get_binary_file_downloader_html(df):
    csv = df.to_csv(index = False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'''
       <a href="data:file/csv;base64,{b64}" download="PredictedHeart.csv">
       <button style="
           background-color:#4CAF50;
           color:white;
           padding:10px 20px;
           border:none;
           border-radius:6px;
           font-size:16px;
           cursor:pointer;">
           Download Predicted CSV
       </button>
       </a>
       '''
    return href

st.markdown("""
<h1 style='
background-color:#0ac1c7;
padding:15px;
border-radius:12px;
'>
❤️ Heart Disease Predictor Dashboard
</h1>
""", unsafe_allow_html=True)
tab1 , tab2 , tab3 = st.tabs(["🔍 Predict" , "📂 Bulk Predict" , "📊 Model Information"])

with tab1:

    age = st.number_input("Age (years)" , min_value = 0 , max_value = 150)
    sex = st.selectbox("Sex" , ["Male" , "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina" , "Atypical Angina" , "Non-Anginal Pain" , "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)" , min_value = 0 , max_value = 300)
    cholestrol = st.number_input("Serum Cholestrol (mg/dl)" , min_value = 0 , max_value = 1000)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl" , ">120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG result" , ["Normal" , "ST-T Wave Abnormality" , "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved" , min_value= 60 , max_value = 202)
    excercise_angina = st.selectbox("Excercise-Angina Induced" , ["Yes" , "No"])
    oldpeak = st.number_input("Oldpeak (St Depression)" , min_value= 0.0 , max_value = 10.0)
    st_slope = st.selectbox("Slope of Peak Excercise ST Segment" , ["Upsloping" , "Flat" , "Downsloping"])

  # convert input into categorial data
    sex = 0 if sex == "Male" else 1
    chest_pain = ["Typical Angina","Atypical Angina","Non-Anginal Pain","Asymptomatic"].index(chest_pain)
    fasting_bs = 1 if fasting_bs=="> 120 mg/dl" else 0
    resting_ecg = ["Normal" , "ST-T Wave Abnormality" , "Left Ventricular Hypertrophy"].index(resting_ecg)
    excercise_angina = 1 if excercise_angina == "Yes" else 0
    st_slope = ["Upsloping" , "Flat" , "Downsloping"].index(st_slope)

# create dataframe for user input

    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholestrol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [excercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })

    algonames = [ "Logistic Regression" ,"Decision Tree", "Random Forest" , "Support Vector Machine"]
    modelnames = [ "heart/Logistic.pkl" ,"heart/DecTree1 (1).pkl ", "heart/Rfc.pkl" , "heart/Svm.pkl"]


    def predict_heart_disease(data):
        predictions = []
        for modelname in modelnames:
            model = pickle.load(open(modelname, 'rb'))
            prediction = model.predict(data)
            predictions.append(prediction)
        return predictions

    if st.button("Submit"):
        st.success("Prediction completed successfully ✅")
        st.subheader('Results.....')
        st.markdown('----------------------------------')

        result = predict_heart_disease(input_data)
        for i in range(len(result)):
            st.subheader(algonames[i])
            agreement= sum([p[0] for p in result])

            if agreement >= 2:

                st.markdown("""
                <div style="
                background-color:#ffe6e6;
                padding:20px;
                border-radius:14px;
                border-left:8px solid #d63031;
                font-size:20px;
                color:#7a1f1f;
                font-weight:bold;
                ">
                ⚠ Final Decision: Heart Disease Likely
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown("""
                <div style="
                background-color:#e8f7ee;
                padding:20px;
                border-radius:14px;
                border-left:8px solid #2ecc71;
                font-size:20px;
                color:#1b5e20;
                font-weight:bold;
                ">
                ✅ Final Decision: No Heart Disease Likely
                </div>
                """, unsafe_allow_html=True)

            st.markdown('------------------------')

with tab2:
    st.title("Upload CSV File")

    st.markdown("""
    <div style="
    background-color:#c8c8f7;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.05);
    ">

    <h3 style="color:#2c3e50;">
    📂 CSV Upload Instructions
    </h3>

    <p style="font-size:15px; color:#0b0b40;">
    Please upload a CSV file using the exact column names and values listed below.
    Incorrect formatting may cause prediction errors.
    </p>

    <hr>

    <h4 style="color:#4a6fa5;">Required Columns</h4>

    <p style = "color:#0b0b40">
    Age, Sex, ChestPainType, RestingBP, Cholesterol,  
    FastingBS, RestingECG, MaxHR, ExerciseAngina,  
    Oldpeak, ST_Slope
    </p>

    <hr>

    <h4 style="color:#4a6fa5;">Allowed Values</h4>
    
    <p style = "color:#0b0b40">
    <b>Sex</b><br>
    Male / Female
    </p>
    <br><br>
    <p style = "color:#0b0b40">
    <b>ChestPainType</b><br>
    Typical Angina / Atypical Angina / Non-Anginal Pain / Asymptomatic
    </p>
    <br><br>
    <p style = "color:#0b0b40">
    <b>FastingBS</b><br>
    <= 120 mg/dl / >120 mg/dl
    </p>
    <br><br>
    <p style = "color:#0b0b40">
    <b>RestingECG</b><br>
    Normal / ST-T Wave Abnormality / Left Ventricular Hypertrophy
    </p>
    <br><br>
    <p style = "color:#0b0b40">
    <b>ExerciseAngina</b><br>
    Yes / No
    </p>
    <br><br>
    <p style = "color:#0b0b40">
    <b>ST_Slope</b><br>
    Upsloping / Flat / Downsloping
    </p>
    <hr>

    <h4 style="color:#4a6fa5;">Example CSV Row</h4>
    <p style = "color:#0b0b40">
    45, Male, Typical Angina, 120, 230, <= 120 mg/dl, Normal, 150, No, 1.2, Flat
    </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a CSV file" , type =["csv"])

    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        model = pickle.load(open('heart/Logistic.pkl' , 'rb'))

        expected_columns = [
            'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
            'Oldpeak', 'ST_Slope'
        ]

        if set(expected_columns).issubset(input_data.columns):

            model = pickle.load(open("heart/Logistic.pkl", "rb"))
            input_data['Sex'] = input_data['Sex'].map({'M': 1, 'F': 0})

            input_data['ChestPainType'] = input_data['ChestPainType'].map({
                'TA': 0,
                'ATA': 1,
                'NAP': 2,
                'ASY': 3
            })

            input_data['RestingECG'] = input_data['RestingECG'].map({
                'Normal': 0,
                'ST': 1,
                'LVH': 2
            })

            input_data['ExerciseAngina'] = input_data['ExerciseAngina'].map({
                'Y': 1,
                'N': 0
            })

            input_data['ST_Slope'] = input_data['ST_Slope'].map({
                'Up': 0,
                'Flat': 1,
                'Down': 2
            })
            input_data['Prediction LR'] = model.predict(input_data[expected_columns])

            input_data.to_csv('PredictedHeartLR.csv', index=False)

            st.subheader("Predictions")
            st.write(input_data)
            positive_cases = input_data['Prediction LR'].sum()

            total_cases = len(input_data)

            st.markdown(f"""
            <div style="
            background-color:#dbe9ff;
            padding:22px;
            border-radius:15px;
            border-left:8px solid #2c5aa0;
            font-size:18px;
            color:#1f3b73;
            box-shadow:0px 4px 10px rgba(0,0,0,0.08);
            margin-bottom:20px;
            ">
            📊 <b>Summary Report</b><br><br>

            Total Records: <b>{total_cases}</b><br>
            Heart Disease Detected: <b>{positive_cases}</b><br>
            Healthy Patients: <b>{total_cases - positive_cases}</b>

            </div>
            """, unsafe_allow_html=True)

            st.download_button(
                label="Download Predicted CSV",
                data=input_data.to_csv(index=False),
                file_name="PredictedHeart.csv",
                mime="text/csv"
            )
        else:
            st.warning("Please make sure the uploaded file has the same columns.")
    else:
        st.info("Upload a CSV file to get the predictions.")

import plotly.express as px
with tab3:

    data = {"Logistic Regression" : 85.8695 ,"Decision Tree" : 80.97826, "Random Forest" : 86.41304, "Support Vector Machine" : 84.2292}
    Models = list(data.keys())
    Accuracies = list(data.values())

    df = pd.DataFrame(list(zip(Models,Accuracies)), columns =['Models','Accuracies'])
    fig = px.bar(
        df,
        y='Accuracies',
        x='Models',
        color='Models',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig)







