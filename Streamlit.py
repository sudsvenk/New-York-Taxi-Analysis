import streamlit as st
import pandas as pd
import pickle
import time
import webbrowser
#import numpy as np

url1 = 'https://www.linkedin.com/in/sudarshan2020/'
url2 = 'https://github.com/sudsvenk/New-York-Taxi-Analysis'

st.set_page_config(
    page_title="Machine Learning App",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "# Hey! Welcome to my Web-App in which I show the Ml Model I built!"
    }
)
header = st.container()
#features = st.container()
#dataset = st.container()

tab0, tab1, tab2, tab3, tab4 = st.tabs(['Introduction','Run Model','Sources','Visualization','More Info'])

pickle_in = open("random_regressor_final.pkl","rb")
random_regressor = pickle.load(pickle_in)

@st.cache
def get_data(filename):
	taxi_data = pd.read_csv(filename)

	return taxi_data

with st.sidebar:
    st.header('Useful Links')
    st.markdown('Connect to me on LinkedIn')
    if st.button('LinkedIn'):
        webbrowser.open_new_tab(url1)
    st.markdown('Check out my project on Github')
    if st.button('Github'):
        webbrowser.open_new_tab(url2)

# Using "with" notation
#with st.sidebar:
#    add_radio = st.radio(
#        "Choose a shipping method",
#        ("Standard (5-15 days)", "Express (2-5 days)")
#    )

with header:
    #picture = st.camera_input("Take a picture")

    #if picture:
     #   st.image(picture)
    #st.title("NYC Cab Price Prediction Analysis")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit NYC Cab Price Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.subheader("Predict how much money or the average money that people spend for a cab ride in new york in certain region in given hour of a day of a month.")

with tab0:
    st.subheader('Objective')
    st.markdown('The goal of this project is to build a machine learning model that can accurately predict the average amount of money that people spend on cab rides in a certain region of New York at a given time of day.')
    st.markdown('This problem is a supervised regression problem. Supervised because we have the actual value of the value we’re trying to predict and regression because what we’re trying to predict is a continuous variable (as opposed to categorical).')
    st.subheader('Methodology')
    st.markdown('To accomplish this goal, we will gather data on past cab rides in New York, including the location, time of day, and fare paid. We will then use this data to train and test a machine learning model that can predict the average fare for a cab ride in a given region at a given time of day.')
    st.subheader('Expected Results')
    st.markdown('We expect that our machine learning model will be able to accurately predict the average fare for a cab ride in a given region at a given time of day, based on the data we have collected. This will allow us to provide more accurate estimates of cab fares to potential riders, and may also be useful for cab companies in pricing their services.')
    st.subheader('Potential Applications')
    st.markdown('The ability to accurately predict cab fares could be useful for a variety of stakeholders, including cab companies, riders, and transportation planners. By providing more accurate fare estimates, we can help cab companies to price their services more effectively and riders to better plan their transportation needs.')
    with st.expander('Data Problem I fixed'):
        st.write("Data Problem 1: Taking care of the negative values in 'total_amount' feature: ")
        st.image('images/Negative_and_zero_values_graph.png')
        st.markdown('Negative total_amount values: removed them since they are likely faulty data points and there weren’t many of them Zero values for total_amount: Same with negative values. Zero values are removed.')
        st.write('Data Problem 2: High Values/Outliers: ')
        st.image('images/too_high_values_graph.png')
        st.markdown('Too-high values for total_amount: some values for total_amount were too high, going as high as 600000. As these are unlikely values for a taxi fare, I decided to come up with an upper limit. The average taxi_fare was approx 16 dollars and there were only 1166 data points higher than 200 dollars. Compared to the 7667792 data points, this is not a great loss of information. Thus, I decided to remove data points with a total_amount value higher than 200.')
    with st.expander('Feature Engineering'):
        st.write('I have conducted some feature engineering on the dataset. Created new feature, Deleted a few unnecessary features.')
        st.subheader('Original features from the dataset I have retained for model training:') 
        st.markdown('[‘PULocationID’, ‘transaction_date’,’ transaction_month’,’ transaction_day’, ‘transaction_hour’, ‘trip_distance’,’ total_amount’, ‘count_of_transactions’]')
with tab1:
    
    def nyc_cab_analysis(PULocationID, Borough, transaction_hour, transaction_day, transaction_month):
        
        if Borough == 'EWR':
            Borough = 0.0
        elif Borough == 'Queens':
            Borough = 1.0
        elif Borough == 'Bronx':
            Borough = 2.0
        elif Borough == 'Manhattan':
            Borough = 3.0
        elif Borough == 'Staten Island':
            Borough = 4.0
        elif Borough == 'Brooklyn':
            Borough = 5.0
        prediction=random_regressor.predict([[PULocationID, Borough, transaction_hour, transaction_day, transaction_month]])
        print(prediction)
        return prediction

    def main():
        #st.title("NYC Cab Price Prediction Analysis")
        #html_temp = """
        #<div style="background-color:tomato;padding:10px">
        #<h2 style="color:white;text-align:center;">Streamlit NYC Cab Price Prediction ML App </h2>
        #</div>
        #"""
        #st.markdown(html_temp,unsafe_allow_html=True)
        #PULocationID = st.text_input("Enter the Pick Up Location ID please: (0-263) ","Type Here")
        st.header('Please give set inputs for the Machine Learning Model to give you prediction!')
        PULocationID = st.slider('Enter the Pick Up Location ID please:', min_value=0, max_value=263)
        Borough = st.radio('Borough',['Bronx','EWR','Manhattan','Queens','Staten Island','Brooklyn'])
        #transaction_hour = st.text_input("What time of the day are you planning to travel?: ","Type Here")
        transaction_hour = st.slider('What hour of the day are you planning to travel?:', min_value=1, max_value=24)
        #transaction_day = st.text_input("Which day of the month are you planning to travel?: ","Type Here")
        transaction_day = st.slider('Which day of the month are you planning to travel?:', min_value=1, max_value=31)
        #transaction_month = st.text_input("Which month of the year are you planning to travel?: ","Type Here")
        transaction_month = st.slider('Which month of the year are you planning to travel?:', min_value=1, max_value=12)
        result=""
        if st.button("Predict"):
            result = nyc_cab_analysis(PULocationID, Borough, transaction_hour, transaction_day, transaction_month)
            with st.spinner('Wait for it...'):
                time.sleep(5)
            st.write('The inputs you have selected are: Pick-Up Location ',PULocationID,'Borough ',Borough,' Hour ',transaction_hour,'day ',transaction_day,'month ', transaction_month)
            
            st.success('The Total Amount is {} dollars'.format(result))
            st.balloons()
    if __name__=='__main__':
        main()
with tab2:
    st.header('This Page shows the Sources from which I downloaded the Dataset Required for the project')
    st.subheader('Sources:')
    st.subheader('1. NYC Yellow taxi dataset')
    st.markdown('- https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page')
    st.subheader('2. Data Dictionary for the yellow cab in NYC')
    st.markdown('- https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf')

with tab3:
        taxi_data = get_data('data/yellow_tripdata_2019-01.csv')
    	# st.write(taxi_data.head())
        st.subheader('Pick-up location ID distribution on the NYC dataset')
        pulocation_dist = pd.DataFrame(taxi_data['PULocationID'].value_counts()).head(50)
        st.bar_chart(pulocation_dist)
        
        st.subheader('Drop-off distribution on the NYC dataset')
        drop_off = pd.DataFrame(taxi_data['DOLocationID'].value_counts()).head(50)
        st.bar_chart(drop_off)
        
        st.subheader('Passenger count distribution on the NYC dataset')
        passenger_count = pd.DataFrame(taxi_data['passenger_count'].value_counts()).head(50)
        st.bar_chart(passenger_count)
        
        st.subheader('Trip distance distribution on the NYC dataset')
        trip_dist = pd.DataFrame(taxi_data['trip_distance'].value_counts()).head(50)
        st.bar_chart(trip_dist)
        
        st.subheader('Total Amount distribution on the NYC dataset')
        tot_amt = taxi_data[(taxi_data['total_amount']>=0) & (taxi_data['total_amount']<200)]
        total_amt = pd.DataFrame(tot_amt['total_amount'].value_counts()).head(50)
        st.bar_chart(total_amt)

with tab4:
    st.subheader('Taxi Zones')
    with st.expander('Bronx'):
        st.markdown('Pick-up Locations IDs in Bronx')
        st.image('images/taxi_zone_map_bronx.jpg')
    with st.expander('Brooklyn'):
        st.markdown('Pick-up Locations IDs in Brooklyn')
        st.image('images/taxi_zone_map_brooklyn.jpg')
    with st.expander('Manhattan'):
        st.markdown('Pick-up Locations IDs in Manhattan')
        st.image('images/taxi_zone_map_manhattan.jpg')
    with st.expander('Queens'):
        st.markdown('Pick-up Location IDs in Queens')
        st.image('images/taxi_zone_map_queens.jpg')
    with st.expander('Staten Island'):
        st.markdown('Pick-up Location IDs in Staten Island')
        st.image('images/taxi_zone_map_staten_island.jpg')