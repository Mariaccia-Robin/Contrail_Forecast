# -*- coding: utf-8 -*-
"""
Created on Mon May  5 11:07:35 2025

@author: Hood
"""

import argparse
import pandas as pd
from inference.inference import predict 

def main(input_path, output_path):
    
    raw_input = pd.read_csv(input_path)
    output = predict(raw_input)
    output.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Predict contrail forecast")
    
    # Adding arguments for input and output file paths with default values
    parser.add_argument(
        "input_path", 
        type=str, 
        nargs="?",  # Make the argument optional
        default="./data.csv",  # Default input path
        help="Path to the input data CSV file (default: './data.csv')"
    )
    parser.add_argument(
        "output_path", 
        type=str, 
        nargs="?",  # Make the argument optional
        default="./output.csv",  # Default output path
        help="Path where the output CSV file should be saved (default: './output.csv')"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the main function with the provided or default arguments
    main(args.input_path, args.output_path)