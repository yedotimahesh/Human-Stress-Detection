import pickle
import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

class StressDetectorApp:
    def __init__(self):
        # Load the pre-trained machine learning model
        self.loaded_model = pickle.load(open('random_forest_model.sav', 'rb'))
        st.set_page_config(
            page_title="Stress Detector",
            page_icon="😃",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        self.create_sidebar()

    def create_sidebar(self):
        st.sidebar.header("About")
        st.sidebar.write("This app predicts stress levels based on sleep-related data.")

    def run(self):
        self.create_main_content()

    def create_main_content(self):
        st.title("Human Stress Detection in and Through Sleep")
        st.subheader("An Interactive Web Application")
        st.image("stress.jpg", use_column_width=True)

        self.get_user_input()

    def get_user_input(self):
        st.header("Enter Sleep Data")
        st.write("Please provide the following sleep-related information:")

        sr = st.slider('Snoring Rate (db)', 45, 100, 71)
        rr = st.slider('Respiration Rate (breaths per minute)', 15, 30, 21)
        t = st.slider('Body Temperature (F)', 85, 100, 92)
        lm = st.slider('Limb Movement', 4, 20, 11)
        bo = st.slider('Blood Oxygen', 80, 100, 90)
        rem = st.slider('Eye Movement', 60, 105, 88)
        sh = st.slider('Sleeping Hours (hr)', 0, 12, 4)
        hr = st.slider('Heart Rate (bpm)', 50, 100, 64)

        if st.button("Predict Stress Level"):
            with st.spinner(text='In Progress'):
                predictions = self.predictor(sr, rr, t, lm, bo, rem, sh, hr)
                stress_level = int(predictions[0])
                stress_category = self.level(stress_level)
                st.success(f"Predicted Stress Level: {stress_category}")
                self.Precautions(stress_category)
                self.display_data_summary()
                self.display_model_information()
                self.display_visualization(stress_level)

    def predictor(self, sr, rr, t, lm, bo, rem, sh, hr):
        prediction = self.loaded_model.predict([[sr, rr, t, lm, bo, rem, sh, hr]])
        return prediction

    def level(self, n):
        if n == 0:
            return "Low / Normal"

        elif n == 1:
            return "Medium Low"
        elif n == 2:
            return "Medium"
        elif n == 3:
            return "Medium High"
        elif n == 4:
            return "High"

    def display_data_summary(self):
        st.header("Data Summary")
        df = pd.read_csv("stress.csv")
        st.write(df.head())

    def display_model_information(self):
        st.subheader("Model Information")
        st.write("Algorithm used: Decision Tree")

        st.subheader("Model Training Accuracy")
        st.write("1.00")

    def display_visualization(self, stress_level):
        if stress_level == 0:
            st.subheader("Visualization for Low / Normal Stress Level")
        elif stress_level == 1:
            st.subheader("Visualization for Medium Low Stress Level")
        elif stress_level == 2:
            st.subheader("Visualization for Medium Stress Level")
        elif stress_level == 3:
            st.subheader("Visualization for Medium High Stress Level")
        elif stress_level == 4:
            st.subheader("Visualization for High Stress Level")

    def Precautions(self,stress_level:str):
        if stress_level == "Low / Normal":
            st.markdown("# Low / Normal")
        elif(stress_level=="Medium Low"):
            st.markdown("Medium Low")
        elif(stress_level=="Medium"):
            st.markdown("Medium")

        elif (stress_level == "Medium High"):
            st.markdown("Medium High")
        else:
            st.markdown("High")




            

if __name__ == "__main__":
    app = StressDetectorApp()
    app.run()
