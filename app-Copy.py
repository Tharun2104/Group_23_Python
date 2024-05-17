import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# Load models
models = {
    "logistic_regression": pickle.load(open("logistic_regression.pkl", "rb")),
    "random_forest": pickle.load(open("random_forest.pkl", "rb")),
    "decision_tree": pickle.load(open("decision_tree.pkl", "rb"))
}

st.set_page_config(page_title="Airline Passenger Satisfaction Prediction App", layout="wide")
st.title("Airline Passenger Satisfaction Prediction App")
st.markdown("This application uses machine learning to predict airline passenger satisfaction.")

# Main input section
with st.form("prediction_form"):
    st.header("User Input Parameters")
    Age = st.number_input('Age', min_value=1, max_value=100, value=25)
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    Customer_Type = st.selectbox('Customer Type', ['Loyal Customer', 'Disloyal Customer'])
    Travel_Type = st.selectbox('Type of Travel', ['Business travel', 'Personal Travel'])
    Class = st.selectbox('Class', ['Eco', 'Eco Plus', 'Business'])
    Flight_Distance = st.number_input('Flight Distance', value=500.0, format="%.1f")
    departure_delay_in_minutes = st.number_input('departure_delay_in_minutes', min_value=0, value=0)
    # Arrival_Delay_in_Minutes = st.number_input('Arrival Delay (in minutes)', min_value=0, value=0)
    Cleanliness = st.number_input('Cleanliness', min_value=0, value=0)
    Departure_Arrival_time_convenient = st.number_input('Departure_Arrival_time_convenient', min_value=0, value=0)
    Ease_of_Online_booking = st.number_input('Ease_of_Online_booking', min_value=0, value=0)
    submit_button = st.form_submit_button("Predict")

# Model and help info in the sidebar
st.sidebar.header("Model and Help")
model_choice = st.sidebar.selectbox('Choose a model:', list(models.keys()))
st.sidebar.info("""
If you have any questions or need further assistance, please reach out to our customer care team:
- 📞 **Contact Number:** +1 (555) 123-4567
- 📧 **Email:** support@airsatisfactionapp.com
""")

# Top 10 Airlines
top_airlines = pd.DataFrame({
    "Name": ["Airline A", "Airline B", "Airline C", "Airline D", "Airline E",
             "Airline F", "Airline G", "Airline H", "Airline I", "Airline J"],
    "Country": ["USA", "Canada", "UK", "Germany", "France",
                "Spain", "Italy", "Australia", "Japan", "South Korea"]
})

st.sidebar.header("Top 10 Airlines")
st.sidebar.table(top_airlines)

# Plotting section if prediction is submitted
if submit_button:
    input_data = {
        'Age': Age,
        'Flight_Distance': Flight_Distance,
        'departure_delay_in_minutes': departure_delay_in_minutes,
        # 'Arrival_Delay_in_Minutes': Arrival_Delay_in_Minutes,
        'Type_of_Travel_Business travel': 1 if Travel_Type == 'Business travel' else 0,
        'Type_of_Travel_Personal Travel': 1 if Travel_Type == 'Personal Travel' else 0,
        'Class_Eco': 1 if Class == 'Eco' else 0,
        'Class_Business': 1 if Class == 'Business' else 0,
        'CustomerType_disloyal Customer': 1 if Customer_Type == 'Disloyal Customer' else 0
    }
    # Ensure all training features are included in input_data with 0s if not applicable
    input_df = pd.DataFrame([input_data])

    # Predict using the selected model
    model = models[model_choice]
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Display Prediction Results
    st.subheader('Prediction Results')
    st.write('Predicted Satisfaction:', 'Satisfied' if prediction[0] == 1 else 'Not Satisfied')

    # Health advice based on the prediction
    if prediction[0] == 1:
        st.success("The passenger is predicted to be satisfied with the airline service.")
    else:
        st.error("The passenger is predicted to be dissatisfied with the airline service.")

    # Trend visualization
    st.subheader("Passenger Satisfaction Trend Analysis")
    trend_data = pd.DataFrame({
        "Year": [2019, 2020, 2021, 2022],
        "Satisfaction Rate": [80, 75, 85, 90]
    })
    fig = px.line(trend_data, x='Year', y='Satisfaction Rate', title='Passenger Satisfaction Trends Over the Years')
    st.plotly_chart(fig)

    # Common issues visualization
    issues_data = pd.DataFrame({
        "Issue": ["Late Flight", "Poor Service", "Comfort", "Baggage"],
        "Frequency": [200, 150, 120, 90]
    })
    fig2 = px.bar(issues_data, x='Issue', y='Frequency', title='Common Passenger Issues')
    st.plotly_chart(fig2)
