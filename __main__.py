import os
import argparse
import pandas as pd
from inference.inference import predict  

def main(input_path, output_path):
    """
    Main function to read input data, make predictions, and save the output to a CSV file.
    
    Args:
        input_path (str): Path to the input data CSV file.
        output_path (str): Path to the output CSV file where predictions will be saved.
    """
    # Read the input data from the provided path
    raw_input = pd.read_csv(input_path)
    
    # Make predictions using the model
    output = predict(raw_input)
    
    # Save the output with predictions to the specified path
    output.to_csv(output_path, index=False)

if __name__ == "__main__":
    """
    Entry point for running the prediction pipeline from command line.
    Parses arguments for input and output file paths, and calls the main function.
    """
    
    # Change the working directory to the location of the script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Set default paths for input and output data
    base_path = os.path.dirname(__file__)
    default_in = os.path.join(base_path, "data/data.csv")
    default_out = os.path.join(base_path, "data/output.csv")  

    # Set up argument parser to handle command-line arguments
    parser = argparse.ArgumentParser(description="Predict contrail forecast")
    
    parser.add_argument(
        "input_path", 
        type=str, 
        nargs="?", 
        default=default_in, 
        help=f"Path to the input data CSV file (default: '{default_in}')"
    )
    parser.add_argument(
        "output_path", 
        type=str, 
        nargs="?",
        default=default_out,  
        help=f"Path where the output CSV file should be saved (default: '{default_out}')"
    )
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    main(args.input_path, args.output_path)
