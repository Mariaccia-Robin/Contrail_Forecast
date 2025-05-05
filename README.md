# Contrail Forecast

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Mariaccia-Robin/Contrail_Forecast.git
   ```

2. Create and activate the virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r Contrail_Forecast/requirements.txt
   ```

## Model Training

- The model is trained in the `notebooks/eda_and_model_training.ipynb` file.
- Follow the instructions in the notebook to preprocess the data, train the model, and save it.

## Inference
- Before running the inference, you need to add the dataset files (.grib) I sent with the rest of my deliverables. (they were too big to push on git)
Note : I had to download the datasets because I couldn't get the ERA5 APIs to work, as I had very limited time to complete the project, I had to make concessions.
- To run inference, execute the following Python script:
  ```bash
  python Contrail_Forecast input_file_path output_file_path
  ```
Note: If you use relative paths, the working directory will be the root directory of the project.

## Output

The output file will be a copy of your input file with a new column "Forecasted Contrails (kgCO2e)"

## Architecture

The inference system is designed to run at any time, given access to the Aircraft/Flight/Airport data as well as connection to the two ERA5 datasets APIs
