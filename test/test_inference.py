# -*- coding: utf-8 -*-
"""
Created on Sun May  4 20:34:14 2025

Test script for verifying the functionality of the contrail prediction model.
"""
import os
import pandas as pd
from inference.inference import predict

def test_predict():
    """
    Test the prediction functionality by loading a sample input file, making predictions,
    and asserting that the mean absolute value of the 'Forecasted Contrail (kgCO2e)'
    column is greater than zero.
    
    This test verifies that the model produces non-zero predictions for contrail forecast.
    """
    # Define the base path for data files
    base_path = os.path.dirname(__file__)
    
    # Load the sample data for prediction
    sample = pd.read_csv(os.path.join(base_path, "../data/test_data.csv"))
    
    # Get the prediction output using the model
    out = predict(sample)
    
    # Assert that the mean absolute value of 'Forecasted Contrail (kgCO2e)' is greater than 0 just to know it sucessfully completed
    assert out['Forecasted Contrails (kgCO2e)'].abs().mean() > 0
