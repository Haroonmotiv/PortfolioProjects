#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Check if the dataset exists before loading it
data_file = "bengaluru_house_prices.csv"
if not os.path.exists(data_file):
    st.error(f"Data file '{data_file}' not found.")
else:
    # Load the dataset
    df1 = pd.read_csv(data_file)

    # Preprocessing steps as defined in your original script
    df1 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
    df1.dropna(inplace=True)

    df1['bhk'] = df1['size'].apply(lambda x: int(x.split(' ')[0]))
    
    def is_float(x):
        try:
            float(x)
        except ValueError:
            return False
        return True

    def convert_sqft_to_num(x):
        tokens = x.split('-')
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2
        try:
            return float(x)
        except ValueError:
            return None 

    df1['total_sqft'] = df1['total_sqft'].apply(convert_sqft_to_num)
    df1.dropna(subset=['total_sqft'], inplace=True)
    df1['price_per_sqft'] = df1['price'] * 100000 / df1['total_sqft']

    # Handling locations and one-hot encoding
    df1['location'] = df1['location'].apply(lambda x: x.strip())
    location_stats = df1['location'].value_counts()
    location_stats_less_than_10 = location_stats[location_stats <= 10]
    df1['location'] = df1['location'].apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

    df1 = df1[~(df1.total_sqft / df1.bhk < 300)]

    def remove_pps_outliers(df):
        df_out = pd.DataFrame()
        for key, subdf in df.groupby('location'):
            m = np.mean(subdf.price_per_sqft)
            stdev = np.std(subdf.price_per_sqft)
            reduced_df = subdf[(subdf.price_per_sqft > (m - stdev)) & (subdf.price_per_sqft <= (m + stdev))]
            df_out = pd.concat([df_out, reduced_df], ignore_index=True)
        return df_out
    
    df1 = remove_pps_outliers(df1)

    df1 = df1[df1.bath < df1.bhk + 2]
    df1.drop(['size', 'price_per_sqft'], axis='columns', inplace=True)

    # One-hot encoding for location
    dummies = pd.get_dummies(df1['location'], drop_first=True)
    df1 = pd.concat([df1, dummies], axis='columns')
    df1.drop('location', axis='columns', inplace=True)

    # Prepare feature and target variables
    X = df1.drop(['price'], axis='columns')
    y = df1['price']

    # Load or train the model
    model_file = 'real_estate_model.pkl'
    columns_file = 'columns.pkl'

    if not os.path.exists(model_file) or not os.path.exists(columns_file):
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
        lr_clf = LinearRegression()
        lr_clf.fit(X_train, y_train)

        # Save the model
        with open(model_file, 'wb') as model_file:
            pickle.dump(lr_clf, model_file)
        
        # Save column names
        columns = X.columns.to_list()
        with open(columns_file, 'wb') as columns_file:
            pickle.dump(columns, columns_file)
        
        st.success("Model trained and saved successfully!")
    else:
        # Load the model and column names
        with open(model_file, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(columns_file, 'rb') as columns_file:
            columns = pickle.load(columns_file)

    # Prediction function
    def predict_price(location, sqft, bath, bhk):
        loc_index = np.where(np.array(columns) == location)[0][0]
        x = np.zeros(len(columns))
        x[0] = sqft
        x[1] = bath
        x[2] = bhk
        if loc_index >= 0:
            x[loc_index] = 1
        return model.predict([x])[0]

    # Streamlit app UI
    st.title("Bengaluru House Price Prediction")

    location = st.selectbox("Select Location", sorted([col for col in columns if col not in ['sqft', 'bath', 'bhk']]))
    sqft = st.number_input("Enter total square feet area", min_value=500)
    bath = st.number_input("Enter number of bathrooms", min_value=1)
    bhk = st.number_input("Enter number of BHK", min_value=1)

    if st.button("Predict Price"):
        if sqft > 0 and bath > 0 and bhk > 0:
            prediction = predict_price(location, sqft, bath, bhk)
            st.success(f"The estimated price is {round(prediction, 2)} Lakh INR")
        else:
            st.error("Please enter valid inputs.")
