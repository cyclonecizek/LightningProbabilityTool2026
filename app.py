import streamlit as st
import numpy as np
import pandas as pd
from prediction import predict, predict_limited, predict_15Z, predict_10Z_updated

def main():

  st.title('Cape Canaveral Lightning Probability Tool')
  

  st.header('Sounding Parameters from 10Z')
  col1, col2 = st.columns(2)
  with col1:
    Thompson_Index = st.number_input('Thompson Index (KI - LI)', 0.0, 60.0, step = 0.1, format= "%.1f")
    RH = st.number_input('700-500mb Average RH', 0, 100, step=1)
  with col2:
    #wind_average = st.slider('1000-700mb Average U-Wind Component', -40.0, 40.0, 0.5)
    wind_direction = st.number_input('1000-700mb Average Wind Direction', 0, 360, step = 1)
    wind_speed = st.number_input('1000-700mb Average Wind Speed in kts', 0.0, 100.0, step= 0.1, format= "%.1f")
  
  if st.button('Probability of Lightning 10Z'):
    #result = predict(np.array([[Thompson_Index, wind_average]]))

    wind_average = wind_speed * np.cos(np.deg2rad(270-wind_direction))

    #result = predict(np.array([[Thompson_Index, wind_average, RH]]))
    #result_limited = predict_limited(np.array([[Thompson_Index, wind_average, RH]]))
    #result_str = str(int(result[0])) + '%'
    #st.header('Version 1.0')
    #st.header(str(int(result[0])) + '%')
    result_10Z = predict_10Z_updated(np.array([[Thompson_Index, wind_average, RH]]))  
    st.header('10Z Output (Version 2.0)')
    st.header(str(int(result_10Z[0])) + '%')





  st.header('Sounding Parameters from 15Z')
  col3, col4 = st.columns(2)
  with col3:
    Thompson_Index_15Z = st.number_input('15Z Thompson Index (KI - LI)', 0.0, 60.0, step = 0.1, format= "%.1f")
    RH_15Z = st.number_input('15Z 700-500mb Average RH', 0, 100, step=1)
    PWAT = st.number_input('PWAT (inches)', 0.00, 5.00, step = 0.01, format = "%.01f")
  with col4:
    #wind_average = st.slider('15Z 1000-700mb Average U-Wind Component', -40.0, 40.0, 0.5)
    wind_direction_15Z = st.number_input('15Z 1000-700mb Average Wind Direction', 0, 360, step = 1)
    wind_speed_15Z = st.number_input('15Z 1000-700mb Average Wind Speed in kts', 0.0, 100.0, step= 0.1, format= "%.1f")
  
  if st.button('Probability of Lightning 15Z'):
    wind_average_15Z = wind_speed_15Z * np.cos(np.deg2rad(270-wind_direction_15Z))
    PWAT_mm = PWAT * 25.4

    result_15Z = predict_15Z(np.array([[Thompson_Index_15Z, wind_average_15Z, PWAT_mm, RH_15Z]]))
    #result_limited = predict_limited(np.array([[Thompson_Index, wind_average, RH]]))
    #result_str = str(int(result[0])) + '%'
    st.header('15Z Output')
    st.header(str(int(result_15Z[0])) + '%')


if __name__=='__main__': 
    main()
  



