# -*- coding: utf-8 -*-
"""
Created on Sun May  4 20:34:14 2025

@author: Hood
"""

import joblib
from utils.preprocessing import preprocess


def classifier_predict(X):
    """
    Uses the classifier model to predict whether a flight will generate a contrail.
    
    Args:
        X (DataFrame): DataFrame with the features to predict.
        
    Returns:
        Index: Indices of flights predicted to generate a contrail.
    """
    model_bundle = joblib.load('models/classifier.joblib')
    classifier = model_bundle['model'] 
    features = model_bundle['features']
    X = X[features]
    y_pred = classifier.predict(X)
    return X.index[y_pred == 1]

def regressor_predict(X):
    """
    Uses the regressor model to forecast the contrail pollution (kgCO2e) for contrail flights.
    
    Args:
        X (DataFrame): DataFrame with features for forecasting contrail pollution.
        
    Returns:
        DataFrame: Updated DataFrame with forecasted contrail pollution values.
    """
    model_bundle = joblib.load('models/regressor.joblib')
    regressor = model_bundle['model'] 
    features = model_bundle['features']
    X = X[features].copy()
    y_pred = regressor.predict(X)
    X['Forecasted Contrails (kgCO2e)'] = y_pred
    return X

def predict(raw_input):
    """
    Predicts contrail generation and the associated pollution for the input data.
    Identifies flights that generate contrails and forecasts the pollution for these flights.
    
    Args:
        raw_input (DataFrame): Input data containing flight information.
        
    Returns:
        DataFrame: Output data with predictions, including forecasted contrail pollution for each flight.
    """
    # Preparing the data
    X = preprocess(raw_input)
    
    print('Inference')
    
    # First we identify flights that will create a contrail
    contrail_flights = classifier_predict(X)
    X_contrail = X.loc[contrail_flights].copy()
    
    # Compute pollution amount for contrail flights
    contrail_impact = regressor_predict(X_contrail)
    
    output = raw_input.merge(contrail_impact[['Distance Flown (km)', 'Forecasted Contrails (kgCO2e)']], 
                             on='Distance Flown (km)', how='left')
    
    # Missing values are those which were identified by the classification as non-contrail generating flights
    output['Forecasted Contrails (kgCO2e)'] = output['Forecasted Contrails (kgCO2e)'].fillna(0)
    
    print('Inference finished.')

    return output
