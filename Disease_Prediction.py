#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 17:30:09 2023

@author: mpcl
"""
import pandas as pnd
import numpy as nmp
import pickle
import streamlit as smt
from PIL import Image as img
from streamlit_option_menu import option_menu

## Loading the save Models

diabetes_model1 = pickle.load(open('/home/mpcl/Desktop/Disease Prediction System/Model_save5', 'rb'))


def welcome():
    return 'welcome you all'


# here, we will define the function which will make the prediction using the    
# data which the user have imported 

def prediction(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    x_sample = nmp.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    prediction = diabetes_model1.predict(x_sample)
    return prediction[0]


# Here, this is the main function in which we will be defining our webpage

def main():
    ## Now , we will give the title to out web page
    smt.title("Diabetes_Prediction")

    # Now, we will be defining some of the frontend elements of our web     
    # page like the colour of background and fonts and font size, the padding and 
    # the text to be displayed    
    html_temp = """ 
    <div style="background-color: #FFFF00; padding: 16px">  
        <h1 style="color: #000000; text-align: center;">Streamlit Diabetes Prediction Classification ML App</h1>  
    </div>  
    """

    # Now, this line will allow us to display the front-end aspects we have   
    # defined in the earlier
    smt.markdown(html_temp, unsafe_allow_html=True)

    # Here, the following lines will create the text boxes in which the user can    
    # enter the data which is required for making the prediction
    Pregnancies = smt.text_input("Pregnancies", "Type Here")
    Glucose = smt.text_input("Glucose", "Type Here")
    BloodPressure = smt.text_input("BloodPressure" , "Type Here")
    SkinThickness = smt.text_input("SkinThickness" , "Type Here")
    Insulin = smt.text_input("Insulin" , "Type Here")
    BMI = smt.text_input("BMI" , "Type Here")
    DiabetesPedigreeFunction =  smt.text_input("DiabetesPedigreeFunction " , "Type Here")
    Age = smt.text_input("Age" , "Type Here")
        

        

    if smt.button("Predict"):  
        result = prediction(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)  
        smt.success('The output of the above is {}'.format(result))

if __name__ == '__main__':  
    main()